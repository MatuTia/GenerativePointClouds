import os

import torch
from matplotlib import pyplot as plt

from generator import StyleTreeGenerator
from post_process import compute_boundary_interior_points

# 9, 27, 33, 45, 47, 91, 111
torch.manual_seed(42)
torch.set_default_dtype(torch.float32)
dir_name = os.path.dirname(__file__)
device = "cuda"

plt.style.use('tableau-colorblind10')
if __name__ == '__main__':
    # Setting
    model_name = "Surface-Dynamic-3-StyleGAN-64"
    epoch = 500

    # Model definition
    model = StyleTreeGenerator(False, False, False, False, device)
    model.to(device)

    # Model load state
    model_name = os.path.join(dir_name, "model", model_name, "generator", f"generator-{epoch}.pt")
    model.load_state_dict(torch.load(model_name))

    # Generate clouds
    style = torch.randn([1, 1, 96], device=device)
    noise = torch.randn_like(style)
    clouds = model.forward(style, [noise]).detach().cpu()

    # Compute child distance
    b, _, d = clouds.shape

    boundary, interior = compute_boundary_interior_points(clouds.squeeze().detach().numpy())

    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(elev=21, azim=16, roll=-80)
    plt.axis("off")

    ax.scatter(boundary[:, 2], boundary[:, 0], boundary[:, 1], s=8)
    ax.scatter(interior[:, 2], interior[:, 0], interior[:, 1], c='lightslategray', alpha=0.15, s=8)

    plt.show()
