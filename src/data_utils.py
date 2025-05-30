import numpy as np
import xarray as xr


# Load and preprocess data
def load_and_preprocess_data():
    "TODO: Time slice variable?"
    print("Starting data load and preprocessing...")
    zarr_ds = xr.open_zarr(store="~/shared-public/mind_the_chl_gap/IO.zarr", consolidated=True)
    zarr_ds = zarr_ds.sel(lat=slice(32, -11.75), lon=slice(42, 101.75))

    all_nan_dates = (
        np.isnan(zarr_ds["CHL_cmes-level3"]).all(dim=["lon", "lat"]).compute()
    )
    zarr_ds = zarr_ds.sel(time=~all_nan_dates)
    zarr_ds = zarr_ds.sortby("time")
    zarr_ds = zarr_ds.sel(time=slice("2019-01-01", "2022-12-31"))
    return zarr_ds


# Prepare data for PINN
def prepare_data_for_pinn(zarr_ds):
    print("Starting data preparation for PINN...")
    variables = [
        "CHL_cmes-level3",
        "air_temp",
        "sst",
        "curr_dir",
        "ug_curr",
        "u_wind",
        "v_wind",
        "v_curr",
    ]
    data = {var: zarr_ds[var].values for var in variables}

    water_mask = ~np.isnan(data["sst"][0])

    for var in variables:
        data[var] = data[var][:, water_mask]
        data[var] = np.nan_to_num(
            data[var],
            nan=np.nanmean(data[var]),
            posinf=np.nanmax(data[var]),
            neginf=np.nanmin(data[var]),
        )
        if var == "CHL_cmes-level3":
            data[var] = np.log(data[var])  # Use log CHL
        mean = np.mean(data[var])
        std = np.std(data[var])
        data[var] = (data[var] - mean) / std
        data[f"{var}_mean"] = mean
        data[f"{var}_std"] = std

    time = zarr_ds.time.values
    lat = zarr_ds.lat.values
    lon = zarr_ds.lon.values
    time_numeric = (time - time[0]).astype("timedelta64[D]").astype(float)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lat_flat = lat_grid.flatten()[water_mask.flatten()]
    lon_flat = lon_grid.flatten()[water_mask.flatten()]

    return data, time_numeric, lat_flat, lon_flat, water_mask
