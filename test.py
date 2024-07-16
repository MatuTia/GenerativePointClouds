import numpy as np
import torch
from matplotlib import pyplot as plt

from generator import StyleTreeGenerator


class PlotUtils:

    def __init__(self, title: str):
        self.ax = plt.figure().add_subplot(projection='3d')
        self.ax.view_init(elev=0, azim=0, roll=0)
        self.ax.set_title(title)
        plt.axis('off')

    def scatter_plot(self, x, y, z):
        self.ax.scatter3D(x, y, z)

    def perimeter_plot(self, x, y, z):
        self.ax.plot3D(x, y, z)

    def plane_plot(self, x, y, z):
        self.ax.plot_surface(x, y, z)

    @staticmethod
    def show_plot():
        plt.show()


if __name__ == '__main__':

    # Default Setting
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)

    # Training-Name
    data_type = "Surface"
    name = 'StyleGAN'
    degree = 64
    epoch = 971
    dir_name = "mmd"

    metrics = np.empty((1000, 3))
    with open(f'model/{data_type}-{name}-{degree}/log.csv', 'r', newline='\n') as f:
        for i, line in enumerate(f.readlines()[1:]):
            line = line[:-1].split(",")
            metrics[i, :] = line

    smaller = np.min(metrics, axis=0)

    for i, metric in enumerate(["Loss", "MMD", "JSD"]):
        ax = plt.figure().add_subplot()
        plt.title(metric)
        plt.plot(metrics[:, i])
        print()
        ax.set_ylim([0, max(metrics[:, i])])
        plt.axhline(smaller[i], color='orange')
        plt.yticks(np.array([0, max(metrics[:, i]), smaller[i]]))
        plt.show()

    device = 'cpu'

    model = torch.load(f'model/{data_type}-{name}-{degree}/{dir_name}/generator-{epoch}.pt')

    ada_in_after = False
    mapping_branching = False
    truncate_style = False

    # gen = TreeGenerator(degree == 32).to(device)
    gen = StyleTreeGenerator(ada_in_after, mapping_branching, truncate_style, degree == 32, device).to(device)
    gen.load_state_dict(model)

    noise = torch.randn((1, 1, 96), device=device)
    noise = noise.repeat(10, 1, 1)
    style = torch.randn((10, 1, 96), device=device)

    # clouds = gen.forward([noise]).cpu().detach().numpy()
    clouds = gen.forward(style, [noise]).cpu().detach().numpy()

    for cloud in clouds:
        plot = PlotUtils(f"{name}-{degree} for best {dir_name} at epoch {epoch}")
        plot.scatter_plot(cloud[:, 0], cloud[:, 2], cloud[:, 1])
        plot.show_plot()
