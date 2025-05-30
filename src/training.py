import numpy as np
import torch
from models import UNet


# Train the UNet model
def train_unet(data, epochs=100, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    features = ["u_wind", "v_wind", "sst", "air_temp"]
    X = np.stack([data[feature] for feature in features], axis=1)
    y = data["CHL_cmes-level3"]

    # Debug: Print the shapes of X and y before reshaping
    print(f"Original X shape: {X.shape}")
    print(f"Original y shape: {y.shape}")

    num_elements = X.shape[2]
    nearest_square = int(np.floor(np.sqrt(num_elements)) ** 2)
    height = int(np.sqrt(nearest_square))
    width = height

    # Trim X and y to the nearest perfect square
    X = X[:, :, :nearest_square]
    y = y[:, :nearest_square]

    # Reshape X and y to match the expected input shape for UNet
    num_samples = X.shape[0]
    num_features = len(features)

    X = X.reshape(num_samples, num_features, height, width)
    y = y.reshape(num_samples, 1, height, width)

    # Debug: Print the shapes of X and y after reshaping
    print(f"Reshaped X shape: {X.shape}")
    print(f"Reshaped y shape: {y.shape}")

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    generator = torch.Generator(device=device)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # Resize the outputs to match the target size
            outputs = nn.functional.interpolate(
                outputs, size=(height, width), mode="bilinear", align_corners=False
            )
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    save_model(model, "unet_model.pth")
    return model
