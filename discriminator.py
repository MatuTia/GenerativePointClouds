import torch


class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        features = [3, 64, 128, 256, 512, 1024]

        self.conv = torch.nn.Sequential()
        for i in range(len(features) - 1):
            self.conv.append(torch.nn.Conv1d(features[i], features[i + 1], kernel_size=1, stride=1))
            self.conv.append(torch.nn.LeakyReLU(0.2))

        self.pooling = torch.nn.MaxPool1d(kernel_size=2048, stride=1)

        features = [1024, 128, 64, 1]
        self.mlp = torch.nn.Sequential()

        for i in range(len(features) - 1):
            self.mlp.append(torch.nn.Linear(features[i], features[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pooling(x).squeeze()
        x = self.mlp(x)
        return x
