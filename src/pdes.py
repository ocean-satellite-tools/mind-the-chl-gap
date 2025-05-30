import torch
import deepxde as dde


def pde(x, y):
    lat, lon, t = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    d2U_dlat2 = dde.grad.hessian(y, x, component=0, i=0, j=0)
    d2U_dlon2 = dde.grad.hessian(y, x, component=0, i=1, j=1)
    d2U_dt2 = dde.grad.hessian(y, x, component=0, i=2, j=2)

    rho = (
        0.1 * torch.sin(lat) * torch.cos(lon) * torch.exp(-0.1 * t)
        + 0.05 * torch.sin(2 * torch.pi * t / 365)
        + (
            0.5 * air_temp_mean
            + -1.0 * sst_mean
            + 0.05 * curr_dir_mean
            + 0.15 * ug_curr_mean
            + 0.4 * u_wind_mean
            + -0.2 * v_wind_mean
            + 0.3 * v_curr_mean
        )
    )

    residual = d2U_dlat2 + d2U_dlon2 + d2U_dt2 - rho
    return residual
