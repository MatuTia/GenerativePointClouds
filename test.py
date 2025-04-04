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
        self.ax.scatter3D(x, y, z, s=6)

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
    name = 'Surface-Dynamic-3-StyleGAN-64'
    epoch = 500
    dir_name = "generator"

    device = 'cpu'

    model = torch.load(f'model/{name}/{dir_name}/generator-{epoch}.pt')

    ada_in_after = False
    mapping_branching = False
    truncate_style = False

    # gen = TreeGenerator(degree == 32).to(device)
    gen = StyleTreeGenerator(ada_in_after, mapping_branching, truncate_style, False, device).to(device)
    gen.load_state_dict(model)

    noise = torch.randn((10, 1, 96), device=device)
    style = torch.randn((10, 1, 96), device=device)

    clouds = gen.forward(style, [noise]).cpu().detach().numpy()

    for cloud in clouds:
        plot = PlotUtils("")
        plot.scatter_plot(cloud[:, 0], cloud[:, 2], cloud[:, 1])
        plot.show_plot()
