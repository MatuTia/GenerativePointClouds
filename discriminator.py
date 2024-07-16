import torch
import torch_geometric


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


class DynamicEdgeDiscriminator(torch.nn.Module):

    def __init__(self, batch_size: int, device: str):
        super(DynamicEdgeDiscriminator, self).__init__()

        features = [3, 64, 128, 256, 512, 1024]
        k_nn = 10

        self.conv = torch.nn.ModuleList()
        for i in range(len(features) - 1):
            self.conv.append(torch_geometric.nn.DynamicEdgeConv(
                torch.nn.Sequential(
                    torch.nn.Linear(2 * features[i], features[i + 1]),
                    torch.nn.LeakyReLU(0.2)),
                k_nn))

        self.pooling = torch.nn.MaxPool1d(kernel_size=2048, stride=1)

        features = [1024, 128, 64, 1]
        self.mlp = torch.nn.Sequential()

        for i in range(len(features) - 1):
            self.mlp.append(torch.nn.Linear(features[i], features[i + 1]))

        self.batch_graph = (torch.ones((batch_size, 2048), dtype=torch.int8, device=device) *
                            torch.arange(0, batch_size, dtype=torch.int8, device=device).reshape(-1, 1))

        self.batch_graph = self.batch_graph.reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        x = x.reshape(-1, 3)

        for conv in self.conv:
            x = conv(x, self.batch_graph)

        x = x.reshape(batch, -1, 1024)
        x = x.transpose(1, 2)

        x = self.pooling(x).squeeze()
        x = self.mlp(x)
        return x
