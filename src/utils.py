import torch
import numpy as np

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def pad_img(img, water_mask):
    pad_length = np.sum(water_mask) - img.shape[0]
    if pad_length > 0:
        img = np.pad(img, (0, pad_length), mode="constant", constant_values=np.nan)
    return img


def data_to_img(data, water_mask, pad=False):
    "Transform data slice or PINN prediction into image"
    if len(data.shape) > 0:
        data = data.flatten()
    if pad:
        data = pad_img(data, water_mask)
    # Create full NaN arrays matching the water_mask shape
    img_grid = np.full(water_mask.shape, np.nan)
    # Assign the chlorophyll values to the water pixels
    img_grid[water_mask] = data[: np.sum(water_mask)]
    # Reshape grids to match the lat/lon dimensions
    img_grid = img_grid.reshape(water_mask.shape)
    return img_grid