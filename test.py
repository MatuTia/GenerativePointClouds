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

    # with open('./model/SurfaceTraining/log.csv', 'r', newline='\n') as f:
    # with open('./model/ChairTraining/log.csv', 'r', newline='\n') as f:
    # with open('./model/ChairTrainingAdaIN/log.csv', 'r', newline='\n') as f:
    # with open('./model/ChairTrainingTreeGAN/log.csv', 'r', newline='\n') as f:
    # with open('./model/ChairTrainingUpsample/log.csv', 'r', newline='\n') as f:
    with open('./model/ChairTrainingStyleTruncated/log.csv', 'r', newline='\n') as f:
        lines = [float(line[:-1]) for line in f.readlines()[1:]]

    smaller = min(lines)
    plt.figure()
    plt.title('Surface Training')
    plt.plot(lines)
    plt.axhline(smaller, color='orange')
    plt.yticks(np.array([0, 1, smaller]))
    plt.show()

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)

    device = 'cpu'

    # TreeGAN
    # model = torch.load(f"model/ChairTrainingTreeGAN/generator-884.pt")
    # model = torch.load(f"model/ChairTrainingTreeGAN/checkpoint.pt")['generator']

    # AdaIn After TreeGCN
    # model = torch.load(f"model/ChairTraining/generator-808.pt")
    # model = torch.load(f"model/ChairTraining/checkpoint.pt")['generator']
    # model = torch.load(f"model/SurfaceTraining/generator-998.pt")
    # model = torch.load(f"model/SurfaceTraining/checkpoint.pt")['generator']

    # AdaIN Before TreeGCN
    # model = torch.load(f"model/ChairTrainingAdaIN/generator-364.pt")
    # model = torch.load(f"model/ChairTrainingAdaIN/checkpoint.pt")['generator']
    # model = torch.load(f"model/ChairTrainingUpsample/generator-972.pt")
    # model = torch.load(f"model/ChairTrainingUpsample/checkpoint.pt")['generator']
    model = torch.load(f"model/ChairTrainingStyleTruncated/generator-499.pt")
    # model = torch.load(f"model/ChairTrainingStyleTruncated/checkpoint.pt")['generator']

    ada_in_after = False
    mapping_branching = True
    truncate_style = True

    # gen = TreeGenerator().to(device)
    gen = StyleTreeGenerator(ada_in_after, mapping_branching, truncate_style, device).to(device)
    gen.load_state_dict(model)

    # First approach
    # noise = torch.randn((10, 1, 96), device=device)
    # style = torch.randn((10, 1, 96), device=device)

    # Fixing noise
    noise = torch.randn((1, 1, 96), device=device)
    noise = noise.repeat(10, 1, 1)
    style = torch.randn((10, 1, 96), device=device)

    # Fixing  style
    # noise = torch.randn((10, 1, 96), device=device)
    # style = torch.randn((1, 1, 96), device=device)
    # style = style.repeat(10, 1, 1)

    # Only noise input
    # noise = torch.randn((10, 1, 96), device=device)

    clouds = gen.forward(style, [noise]).cpu().detach().numpy()
    # clouds = gen.forward([noise]).cpu().detach().numpy()

    for cloud in clouds:
        plot = PlotUtils("Chair")
        plot.scatter_plot(cloud[:, 0], cloud[:, 2], cloud[:, 1])
        plot.show_plot()
