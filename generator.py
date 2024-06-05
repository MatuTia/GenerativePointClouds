from math import sqrt

import torch


class MapBlock(torch.nn.Module):

    def __init__(self, features: list[int], nodes: int, depth: int, degree: list[int], layers_size: list[int]):
        super(MapBlock, self).__init__()
        self.in_features = features[depth]
        self.out_features = features[depth + 1]
        self.degree = degree[depth]
        self.layer_size = layers_size[depth]
        self.degree = degree[depth]

        if self.degree > 1:
            self.branching = torch.nn.Parameter(torch.empty(nodes, self.in_features, self.in_features * self.degree))
            torch.nn.init.xavier_uniform_(self.branching.data, gain=torch.nn.init.calculate_gain('relu'))

        self.layers = torch.nn.ModuleList()

        if self.layer_size > 1:
            for _ in range(self.layer_size - 1):
                self.layers.append(torch.nn.Linear(self.in_features, self.in_features))

        self.layers.append(torch.nn.Linear(self.in_features, self.out_features))

        self.leakyReLU = torch.nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.degree > 1:
            batch_size, _, features = x.shape
            x = (x.unsqueeze(2) @ self.branching).view(batch_size, -1, features)

        for layer in self.layers:
            x = layer(x)
            x = self.leakyReLU(x)

        return x


class PTBlock(torch.nn.Module):

    def __init__(self, features: list[int], nodes: int, depth: int, degrees: list[int], branching: bool,
                 activation: bool):
        self.in_features = features[depth]
        self.out_features = features[depth + 1]
        self.nodes = nodes
        self.degree = degrees[depth]
        self.activation = activation
        self.branching = branching

        super(PTBlock, self).__init__()

        self.weights = torch.nn.ModuleList()
        for i in range(depth + 1):
            self.weights.append(torch.nn.Linear(features[i], self.out_features, bias=False))

        if branching:
            self.branch = torch.nn.Parameter(torch.empty(self.nodes, self.in_features, self.in_features * self.degree))

        support = 10
        self.support = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.in_features * support, bias=False),
            torch.nn.Linear(self.in_features * support, self.out_features, bias=False),
        )

        self.bias = torch.nn.Parameter(torch.empty(1, self.degree, self.out_features))
        self.sigma = torch.nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.branch, gain=torch.nn.init.calculate_gain('relu'))

        out_bound = 1 / sqrt(self.out_features)
        torch.nn.init.uniform_(self.bias, -out_bound, out_bound)

    def forward(self, tree: list[torch.Tensor], style: torch.Tensor) -> list[torch.Tensor]:
        batch_size = tree[0].size(0)

        y = 0
        for x, weight in zip(tree, self.weights):
            y += weight(x).repeat(1, 1, int(self.nodes / weight(x).size(1))).view(batch_size, -1, self.out_features)

        # Branching & K-support
        if self.branching:
            x = self.sigma(tree[-1].unsqueeze(2) @ self.branch).view(batch_size, -1, self.in_features)
            x = self.support(x)
            y = x + torch.Tensor(y).repeat(1, 1, self.degree).view(batch_size, -1, self.out_features)
        else:
            x = self.support(tree[-1])
            y = x + y

        #  Combine elements
        if self.activation:
            y = self.sigma(y + self.bias.repeat(1, self.nodes, 1))

        assert y.size() == style.size()

        #  AdaIN
        var_y, mean_y = torch.var_mean(y, dim=1, unbiased=False, keepdim=True)
        var_style, mean_style = torch.var_mean(style, dim=1, unbiased=False, keepdim=True)

        std_y, std_style = torch.sqrt(var_y + 1e-5), torch.sqrt(var_style + 1e-5)

        normalized_y = (y - mean_y) / std_y
        y = normalized_y * std_style + mean_style

        tree.append(y)

        return tree


class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.mapping = torch.nn.ModuleList()
        self.synthesis = torch.nn.ModuleList()
        degrees = [1, 2, 2, 2, 2, 2, 64]
        layers_size = [4, 2, 1, 1, 1, 1, 1]
        features = [96, 256, 256, 256, 128, 128, 128, 3]

        assert len(degrees) == len(layers_size) == len(features) - 1

        nodes = 1
        for depth in range(len(degrees)):

            # Mapping
            self.mapping.append(MapBlock(features, nodes, depth, degrees, layers_size))

            # Synthesis
            if depth == len(degrees) - 1:
                self.synthesis.append(PTBlock(features, nodes, depth, degrees, True, False))
            else:
                self.synthesis.append(PTBlock(features, nodes, depth, degrees, True, True))

            nodes *= degrees[depth]

    def forward(self, z: torch.Tensor, x: list[torch.Tensor]) -> torch.Tensor:
        for mapping, synthesis in zip(self.mapping, self.synthesis):
            z = mapping(z)
            x = synthesis(x, z)
        return x[-1]
