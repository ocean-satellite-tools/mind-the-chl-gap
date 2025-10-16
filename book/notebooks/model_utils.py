import numpy as np
import xarray as xr
import pandas as pd

def time_series_split(
    data: xr.Dataset,
    num_var,
    cat_var=None,
    split_ratio=(0.7, 0.2, 0.1),
    seed=42,
    X_mean=None,
    X_std=None,
    y_var="y",
    years=None,               # select one year, a list of years, or a slice
    cast_float32=True,
    contiguous_splits=False,
    return_full=False,
    nan_max_frac=0.05,
    verbose=False
):
    """
    Pure-NumPy splitter/normalizer for xarray Dataset (NumPy-backed). 
    Splits time indices randomly into train/val/test. Replaces NaNs with 0s. 
    Normalizes numerical variables only, using either provided or training-set mean/std. 
    Removes days with too many NaNs (> 
    
    Parameters: 
      data: xarray dataset with 'time' dimension 
      years: year(s) to use for training
      num_var: list of numerical variable names (to normalize) 
      cat_var: list of categorical variable names (no normalization) 
      y_var: name of response variable in data. 
      split_ratio: tuple (train, val, test), must sum to 1.0 
      seed: random seed 
      nan_max_frac: maximum percent missing values for response or explanatory variables
      X_mean, X_std: optional mean/std arrays for num_var only (shape = [n_num_vars]) 
      cast_float32 : If True, cast outputs to float32 (good for TF)
      verbose: print out info
      return_full: return X and y
      contiguous_splits: versus random splits
      
    Returns: 
      X, y: full input and response arrays (NumPy arrays) 
      X_train, y_train, X_val, y_val, X_test, y_test: split data X_mean, X_std: mean and std used for normalization
      If return_full=False, X and y are None.
    """
    if cat_var is None:
        cat_var = []
    input_var = list(num_var) + list(cat_var)

    # --- checks
    if "time" not in data.dims:
        raise ValueError("Dataset must contain a 'time' dimension.")
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError("split_ratio must sum to 1.0")
    if "ocean_mask" not in data:
        raise KeyError("Dataset must contain 'ocean_mask' (1=ocean, 0=land).")

    # ---------- subset by year(s) ----------
    if years is not None:
        if isinstance(years, (str, int)):
            data = data.sel(time=str(years))
        elif isinstance(years, slice):
            data = data.sel(time=years)
        else:
            # assume iterable of years (ints/strs)
            ti = pd.DatetimeIndex(np.asarray(data["time"].values))
            yrs = set(int(y) for y in years)
            sel = xr.DataArray(np.isin(ti.year, list(yrs)), coords={"time": data["time"]}, dims=["time"])
            data = data.sel(time=sel)
    if data.sizes.get("time", 0) == 0:
        raise ValueError("No timesteps left after year filtering.")

    # pick a template for broadcasting 2D -> 3D
    template_name = (input_var[0] if input_var else y_var)
    template = data[template_name]

    # ---------- NaN-based time filtering over ocean ----------
    ocean = data["ocean_mask"].astype(bool)
    if "time" not in ocean.dims:
        ocean = ocean.expand_dims({"time": data["time"]}).broadcast_like(template)
    else:
        ocean = ocean.broadcast_like(template)

    spatial_dims = [d for d in ocean.dims if d != "time"]
    ocean_pix_per_t = ocean.sum(dim=spatial_dims)
    nan_thresh_t = nan_max_frac * ocean_pix_per_t

    check_vars = input_var + [y_var]
    valid_times = xr.DataArray(np.ones(data.sizes["time"], dtype=bool), coords={"time": data["time"]}, dims=["time"])

    for v in check_vars:
        if v not in data:
            raise KeyError(f"Variable '{v}' not found in dataset.")
        arr = data[v]
        if "time" not in arr.dims:
            arr = arr.expand_dims({"time": data["time"]}).broadcast_like(template)
        else:
            arr = arr.broadcast_like(template)
        v_nan = xr.apply_ufunc(np.isnan, arr) & ocean
        v_nan_count = v_nan.sum(dim=spatial_dims)
        # Remove days with too many NaNs (> nan_thresh)
        valid_times = valid_times & (v_nan_count < nan_thresh_t)

    before = int(data.sizes["time"])
    data = data.sel(time=valid_times)
    after = int(data.sizes["time"])
    if after == 0:
        raise ValueError("No timesteps left after NaN filtering.")
    if verbose:
        yrs_msg = f" (years={years})" if years is not None else ""
        print(f"[NaN filter]{yrs_msg} kept {after}/{before} days "
              f"(≤ {nan_max_frac*100:.1f}% NaNs over ocean per variable).")

    # ---------- split indices ----------
    T = int(data.sizes["time"])
    n_train = int(split_ratio[0] * T)
    n_val   = int(split_ratio[1] * T)
    n_test  = T - n_train - n_val

    if contiguous_splits:
        train_idx = slice(0, n_train)
        val_idx   = slice(n_train, n_train + n_val)
        test_idx  = slice(n_train + n_val, T)
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(T)
        train_idx = np.sort(perm[:n_train])
        val_idx   = np.sort(perm[n_train:n_train + n_val])
        test_idx  = np.sort(perm[n_train + n_val:])

    # ---------- helpers ----------
    def fetch(var):
        arr = data[var]
        if "time" not in arr.dims:
            arr = arr.expand_dims({"time": data["time"]}).broadcast_like(template)
        else:
            arr = arr.broadcast_like(template)
        arr_np = arr.transpose("time", ...).values
        if cast_float32:
            arr_np = arr_np.astype("float32", copy=False)
        return arr_np

    # stats from training
    if num_var:
        if X_mean is None or X_std is None:
            means, stds = [], []
            for v in num_var:
                a = fetch(v)
                a_tr = a[train_idx]
                means.append(np.nanmean(a_tr, axis=(0, 1, 2)))
                stds.append( np.nanstd( a_tr, axis=(0, 1, 2)))
            X_mean = np.asarray(means, dtype="float32" if cast_float32 else a.dtype)
            X_std  = np.asarray(stds,  dtype="float32" if cast_float32 else a.dtype)
        X_std_safe = np.where(X_std == 0, 1.0, X_std)
    else:
        X_mean = np.array([], dtype="float32" if cast_float32 else float)
        X_std  = np.array([], dtype="float32" if cast_float32 else float)
        X_std_safe = X_std

    def build_split(idx):
        chans = []
        for k, v in enumerate(num_var):
            a = fetch(v)
            a = a[idx]
            a = (a - X_mean[k]) / X_std_safe[k]
            a = np.nan_to_num(a)
            chans.append(a)
        for v in cat_var:
            a = fetch(v)
            a = a[idx]
            a = np.nan_to_num(a)
            chans.append(a)
        if not chans:
            raise ValueError("No input variables provided.")
        return np.stack(chans, axis=-1)

    y_full = data[y_var].transpose("time", ...).values
    if cast_float32:
        y_full = y_full.astype("float32", copy=False)

    def take_y(idx):
        y_s = y_full[idx]
        return np.nan_to_num(y_s)

    # ---------- build splits ----------
    X_train = build_split(train_idx); y_train = take_y(train_idx)
    X_val   = build_split(val_idx);   y_val   = take_y(val_idx)
    X_test  = build_split(test_idx);  y_test  = take_y(test_idx)

    if return_full:
        if contiguous_splits:
            X = np.concatenate([X_train, X_val, X_test], axis=0)
            y = np.concatenate([y_train, y_val, y_test], axis=0)
        else:
            X = build_split(slice(0, T))
            y = take_y(slice(0, T))
    else:
        X = None
        y = None

    return X, y, X_train, y_train, X_val, y_val, X_test, y_test, X_mean, X_std

# Save and load model bundle
import json, zipfile, tempfile, io
from pathlib import Path
import numpy as np
from keras.models import load_model

# Optional dependency (only needed if you pass custom_objects)
try:
    import cloudpickle
except Exception:  # pragma: no cover
    cloudpickle = None

def save_cnn_bundle(path, model, X_mean, X_std, history=None, meta=None, custom_objects=None):
    """
    Save a portable bundle containing:
      - model.keras
      - preproc_stats.npz (X_mean, X_std)
      - history.json (optional)
      - meta.json (optional)
      - custom_objects.pkl (optional; cloudpickled dict of custom losses/metrics)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1) serialize model to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.keras"
        model.save(model_path)

        # 2) serialize stats
        stats_buf = io.BytesIO()
        np.savez(stats_buf, X_mean=X_mean, X_std=X_std)
        stats_bytes = stats_buf.getvalue()

        # 3) optional blobs
        hist_bytes = json.dumps(history or {}).encode("utf-8")
        meta_bytes = json.dumps(meta or {}).encode("utf-8")

        # 4) write zip
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(model_path, arcname="model.keras")
            z.writestr("preproc_stats.npz", stats_bytes)
            z.writestr("history.json", hist_bytes)
            z.writestr("meta.json",    meta_bytes)

            if custom_objects:
                if cloudpickle is None:
                    raise ImportError("cloudpickle is required to bundle custom_objects")
                z.writestr("custom_objects.pkl", cloudpickle.dumps(custom_objects))

    return str(path)

def load_cnn_bundle(path, compile=False):
    """
    Load a bundle created by save_bundle().
    Returns: (model, X_mean, X_std, history:dict, meta:dict, custom_objects:dict|None)
    """
    path = Path(path)
    with zipfile.ZipFile(path, "r") as z, tempfile.TemporaryDirectory() as tmpdir:
        # model
        model_path = Path(tmpdir) / "model.keras"
        with z.open("model.keras") as fsrc, open(model_path, "wb") as fdst:
            fdst.write(fsrc.read())

        # custom_objects (if present)
        custom_objects = None
        if "custom_objects.pkl" in z.namelist():
            if cloudpickle is None:
                raise ImportError("cloudpickle is required to load bundled custom_objects")
            with z.open("custom_objects.pkl") as f:
                custom_objects = cloudpickle.loads(f.read())

        # load model with customs (or none)
        model = load_model(model_path, custom_objects=custom_objects, compile=compile)

        # stats
        with z.open("preproc_stats.npz") as f:
            buf = io.BytesIO(f.read())
            stats = np.load(buf)
            X_mean, X_std = stats["X_mean"], stats["X_std"]

        # json helpers
        def read_json(name):
            try:
                with z.open(name) as f:
                    return json.loads(f.read().decode("utf-8"))
            except KeyError:
                return {}

        history = read_json("history.json")
        meta    = read_json("meta.json")

    return model, X_mean, X_std, history, meta, custom_objects

## PLOTTING

import numpy as np
import matplotlib.pyplot as plt

def predict_and_plot_date(
    data_xr,
    date,                          # "YYYY-MM-DD" or np.datetime64
    model,
    num_var,                       # list of vars to normalize
    cat_var,                       # list of vars not normalized (e.g., ocean_mask, sin/cos time)
    X_mean, X_std,                 # per-channel stats for num_var (shape [len(num_var)])
    y_var="y",
    mask_var="ocean_mask",
    model_type="cnn",              # "cnn" or "tabular"
    cast_float32=True,
    use_percentiles=True, p_lo=5, p_hi=95,
    cmap="viridis"
):
    """
    Build one-sample input from dataset for a specific date, predict, and plot True vs Pred.
    Works with CNN (map→map) and tabular models (flattened pixels).
    """
    # ---- resolve date index
    date64 = np.datetime64(str(date))
    times = np.asarray(data_xr["time"].values)
    idxs = np.where(times == date64)[0]
    if idxs.size == 0:
        raise ValueError(f"Date {date} not found in dataset time coord.")
    t = int(idxs[0])

    # ---- helper to fetch a variable as (H,W) for that date; broadcast 2D to 3D if needed
    # choose a spatial template (first available among inputs or y)
    tmpl_name = (num_var + cat_var + [y_var])[0]
    tmpl = data_xr[tmpl_name]
    def fetch_2d(varname):
        arr = data_xr[varname]
        if "time" in arr.dims:
            arr_t = arr.isel(time=t)
        else:
            arr_t = arr
        arr_t = arr_t.broadcast_like(tmpl.isel(time=t))  # ensure same H,W
        a = arr_t.values
        if cast_float32:
            a = a.astype("float32", copy=False)
        return a  # (H,W)

    # ---- build channels for this date
    num_chans = []
    for k, vn in enumerate(num_var):
        a = fetch_2d(vn)
        # normalize with training stats, then fill NaNs with 0
        a = (a - X_mean[k]) / (1.0 if X_std[k] == 0 else X_std[k])
        a = np.nan_to_num(a)
        num_chans.append(a)

    cat_chans = []
    for vn in cat_var:
        a = fetch_2d(vn)
        a = np.nan_to_num(a)
        cat_chans.append(a)

    if not (num_chans or cat_chans):
        raise ValueError("No input variables provided.")

    # stack to (H,W,C)
    X_map = np.stack(num_chans + cat_chans, axis=-1)
    H, W, C = X_map.shape

    # ---- ground truth map
    y_true = fetch_2d(y_var)

    # ---- predict
    if model_type == "cnn":
        y_pred = model.predict(X_map[np.newaxis, ...], verbose=0)[0]
        if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
            y_pred = y_pred[..., 0]
    elif model_type == "tabular":
        y_pred = model.predict(X_map.reshape(-1, C)).reshape(H, W)
    else:
        raise ValueError("model_type must be 'cnn' or 'tabular'.")

    # ---- mask land to NaN (mask_var==0 → land)
    land = (fetch_2d(mask_var) == 0.0)
    y_true = np.where(land, np.nan, y_true)
    y_pred = np.where(land, np.nan, y_pred)

    # ---- color limits
    if use_percentiles:
        stack = np.concatenate([y_true[~np.isnan(y_true)], y_pred[~np.isnan(y_pred)]]) if np.isfinite(y_true).any() and np.isfinite(y_pred).any() else np.array([])
        vmin, vmax = (np.percentile(stack, p_lo), np.percentile(stack, p_hi)) if stack.size else (None, None)
    else:
        vmin = np.nanmin([y_true, y_pred]); vmax = np.nanmax([y_true, y_pred])

    # --- ensure North is up
    lat = np.array(data_xr.lat.values)
    flip_lat = lat[0] > lat[-1]   # True if lat is descending

    if flip_lat:
        y_true = np.flipud(y_true)
        y_pred = np.flipud(y_pred)

    # extent must be (xmin, xmax, ymin, ymax) with increasing y
    lon_min, lon_max = float(data_xr.lon.min()), float(data_xr.lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    extent = [lon_min, lon_max, lat_min, lat_max]

    # ---- plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].imshow(y_true, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_title(f"True {y_var} — {np.datetime_as_string(date64)}"); axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(y_pred, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_title(f"Predicted ({model_type.upper()})"); axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.show()
    return y_true, y_pred

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_true_vs_predicted_year(data, year, model, X_mean, X_std, num_var, cat_var, y_var="y"):
    data_year = data.sel(time=year)

    # pick first available day each month
    dates = pd.to_datetime(data_year.time.values)
    monthly_dates = (
        pd.Series(dates)
        .groupby([dates.year, dates.month])
        .min()
        .sort_values()
    )
    n_months = len(monthly_dates)

    lat = data_year.lat.values
    lon = data_year.lon.values
    flip_lat = lat[0] > lat[-1]
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    land_mask = (data_year["ocean_mask"].values == 0.0)

    # helper: fetch a 2D array for a var at a given date; broadcast if var has no time dim
    def fetch_2d(var, date):
        arr = data_year[var]
        arr_t = arr.sel(time=date) if "time" in arr.dims else arr
        arr_t = arr_t.broadcast_like(data_year[y_var].sel(time=date))
        a = arr_t.values.astype("float32", copy=False)
        return a

    fig, axs = plt.subplots(n_months, 2, figsize=(7, 2*n_months), constrained_layout=True)

    for i, date in enumerate(monthly_dates):
        # build (H,W,C) for this date
        chans = []
        for k, v in enumerate(num_var):
            a = fetch_2d(v, date)
            denom = 1.0 if X_std[k] == 0 else X_std[k]
            a = (a - X_mean[k]) / denom
            a = np.nan_to_num(a)
            chans.append(a)
        for v in cat_var:
            a = fetch_2d(v, date)
            a = np.nan_to_num(a)
            chans.append(a)
        X_map = np.stack(chans, axis=-1)

        # true & pred
        true_output = fetch_2d(y_var, date)
        pred = model.predict(X_map[np.newaxis, ...], verbose=0)[0]
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred[..., 0]
        predicted_output = pred

        # mask land
        predicted_output[land_mask] = np.nan
        true_output = np.where(land_mask, np.nan, true_output)

        # north-up if lat is descending
        if flip_lat:
            true_output = np.flipud(true_output)
            predicted_output = np.flipud(predicted_output)

        # common color range (robust)
        vmin = np.nanpercentile([true_output, predicted_output], 5)
        vmax = np.nanpercentile([true_output, predicted_output], 95)

        # metrics
        m = ~np.isnan(true_output) & ~np.isnan(predicted_output)
        r2 = r2_score(true_output[m].ravel(), predicted_output[m].ravel()) if m.any() else np.nan
        rmse = np.sqrt(np.nanmean((true_output - predicted_output)**2))

        # plots
        axs[i, 0].imshow(true_output, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='viridis', aspect='equal')
        axs[i, 0].set_title(f"{date.strftime('%b')} — True", fontsize=10); axs[i, 0].axis('off')

        axs[i, 1].imshow(predicted_output, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='viridis', aspect='equal')
        axs[i, 1].set_title(f"{date.strftime('%b')} — Pred\n$R^2$={r2:.2f}, RMSE={rmse:.2f}", fontsize=10); axs[i, 1].axis('off')

    plt.suptitle(f'CHL: True vs Predicted (log scale) {year}', fontsize=16)
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
import calendar

def plot_metric_by_month(
    data, years, model, X_mean, X_std, num_var, cat_var,
    training_year=None, metric='r2',
    y_name='y', mask_var='ocean_mask',
    ssim_win_size=None, ssim_sigma=None
):
    assert metric in ['r2', 'rmse', 'mae', 'bias', 'ssim']

    def fetch_2d(ds, var, date, like_var):
        """Return 2D array for var at date; broadcast if var has no time dim."""
        arr = ds[var]
        arr_t = arr.sel(time=date) if 'time' in arr.dims else arr
        arr_t = arr_t.broadcast_like(ds[like_var].sel(time=date))
        return arr_t.values.astype('float32', copy=False)

    metric_by_year_month = {}

    for year in years:
        ds = data.sel(time=year)
        dates = pd.to_datetime(ds.time.values)
        monthly_dates = (
            pd.Series(dates).groupby([dates.year, dates.month]).min().sort_values()
        )

        scores = []
        for date in monthly_dates:
            # build (H,W,C) input for this date
            chans = []
            for k, v in enumerate(num_var):
                a = fetch_2d(ds, v, date, y_name)
                denom = 1.0 if X_std[k] == 0 else X_std[k]
                a = (a - X_mean[k]) / denom
                chans.append(np.nan_to_num(a))
            for v in cat_var:
                a = fetch_2d(ds, v, date, y_name)
                chans.append(np.nan_to_num(a))
            X_map = np.stack(chans, axis=-1)

            # predict
            pred = model.predict(X_map[np.newaxis, ...], verbose=0)[0]
            if pred.ndim == 3 and pred.shape[-1] == 1:
                pred = pred[..., 0]

            # truth & mask
            truth = fetch_2d(ds, y_name, date, y_name)
            land = (fetch_2d(ds, mask_var, date, y_name) == 0.0)
            pred  = np.where(land, np.nan, pred)
            truth = np.where(land, np.nan, truth)

            # metric
            if metric == 'ssim':
                # fill NaNs for SSIM computation
                t = np.nan_to_num(truth, nan=(np.nanmean(truth) if np.isfinite(truth).any() else 0.0))
                p = np.nan_to_num(pred,  nan=(np.nanmean(pred)  if np.isfinite(pred).any()  else 0.0))
                # robust data_range
                dr = np.nanmax(truth) - np.nanmin(truth)
                if not np.isfinite(dr) or dr == 0:
                    dr = (np.nanmax(t) - np.nanmin(t)) or 1.0
                # build kwargs safely (don’t pass sigma=None)
                ssim_kwargs = {"data_range": dr}
                if ssim_win_size is not None:
                    ssim_kwargs["win_size"] = int(ssim_win_size)  # must be odd
                if ssim_sigma is not None:
                    ssim_kwargs["gaussian_weights"] = True
                    ssim_kwargs["sigma"] = float(ssim_sigma)
                score = ssim(t.astype(np.float64), p.astype(np.float64), **ssim_kwargs)
            else:
                m = ~np.isnan(truth) & ~np.isnan(pred)
                if not m.any():
                    score = np.nan
                elif metric == 'r2':
                    score = r2_score(truth[m].ravel(), pred[m].ravel())
                elif metric == 'rmse':
                    score = float(np.sqrt(np.mean((truth[m] - pred[m])**2)))
                elif metric == 'mae':
                    score = float(mean_absolute_error(truth[m], pred[m]))
                elif metric == 'bias':
                    score = float(np.mean(pred[m] - truth[m]))

            scores.append(score)

        metric_by_year_month[year] = (monthly_dates.dt.month.values, scores)

    # plot
    plt.figure(figsize=(10,5))
    for year, (months, scores) in metric_by_year_month.items():
        label = f"{year} (train)" if year == training_year else year
        style = "--" if year == training_year else "-"
        plt.plot(months, scores, style, marker='o', label=label)

    plt.xlabel("Month")
    plt.ylabel({'r2':"$R^2$",'rmse':"RMSE",'mae':"MAE",'bias':"Bias",'ssim':"SSIM"}[metric])
    plt.title(f"Monthly {metric.upper()} by Year")
    plt.xticks(np.arange(1,13), calendar.month_abbr[1:13])
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()