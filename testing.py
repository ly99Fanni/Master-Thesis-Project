import torch

weights = torch.load(
    "dqn_model.pth",
    map_location="cpu"
)

print(weights)

for name, tensor in weights.items():
    print(
        f"{name:30s} "
        f"shape={tuple(tensor.shape)}"
    )