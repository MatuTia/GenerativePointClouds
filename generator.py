from math import sqrt

import torch


class AdaIN(torch.nn.Module):

    def __init__(self, dim: int):
        super(AdaIN, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        assert x.size() == style.size()

        #  AdaIN
        var_x, mean_x = torch.var_mean(x, dim=self.dim, unbiased=False, keepdim=True)
        var_style, mean_style = torch.var_mean(style, dim=self.dim, unbiased=False, keepdim=True)

        std_x, std_style = torch.sqrt(var_x + 1e-5), torch.sqrt(var_style + 1e-5)

        normalized_x = (x - mean_x) / std_x
        return normalized_x * std_style + mean_style


class TreeGCN(torch.nn.Module):

    def __init__(self, features: list[int], nodes: int, depth: int, degrees: list[int], branching: bool,
                 activation: bool):
        self.in_features = features[depth]
        self.out_features = features[depth + 1]
        self.nodes = nodes
        self.degree = degrees[depth]
        self.activation = activation
        self.branching = branching

        super(TreeGCN, self).__init__()

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

    def forward(self, tree: [torch.Tensor]) -> [torch.Tensor]:
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

        tree.append(y)

        return tree


class MapBlock(torch.nn.Module):

    def __init__(self, features: list[int], nodes: int | None, depth: int, degree: list[int], layers_size: list[int],
                 branch: bool, device: str):
        super(MapBlock, self).__init__()
        self.in_features = features[depth]
        self.out_features = features[depth + 1]
        self.degree = degree[depth]
        self.layer_size = layers_size[depth]
        self.degree = degree[depth]
        self.device = device
        self.branch = branch

        if branch and self.degree > 1:
            assert nodes is not None
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
            if self.branch:
                batch_size, _, features = x.shape
                x = (x.unsqueeze(2) @ self.branching).view(batch_size, -1, features)
            else:
                radius = .2
                batch_size, points, features = x.shape
                rand = torch.rand(batch_size, points, features * (self.degree - 1), device=self.device) * radius
                rand += x.repeat(1, 1, self.degree - 1) - radius / 2
                x = torch.cat([x, rand], dim=2).reshape(batch_size, -1, features)

        for layer in self.layers:
            x = layer(x)
            x = self.leakyReLU(x)

        return x


class PTBlock(torch.nn.Module):

    def __init__(self, features: list[int], nodes: int, depth: int, degrees: list[int], branching: bool,
                 activation: bool, after: bool):
        super(PTBlock, self).__init__()

        self.ada_in = AdaIN(1)
        self.tree_gcn = TreeGCN(features, nodes, depth, degrees, branching, activation)
        self.after = after

    def forward(self, tree: list[torch.Tensor], style: torch.Tensor) -> list[torch.Tensor]:

        if self.after:
            tree = self.tree_gcn.forward(tree)
            tree[-1] = self.ada_in.forward(tree[-1], style)
        else:
            tree[-1] = self.ada_in.forward(tree[-1], style)
            tree = self.tree_gcn.forward(tree)

        return tree


class StyleTreeGenerator(torch.nn.Module):

    def __init__(self, after: bool, mapping_branching: bool, truncate_style: bool, alternative_degrees: bool,
                 device: str):
        super(StyleTreeGenerator, self).__init__()

        self.mapping = torch.nn.ModuleList()
        self.synthesis = torch.nn.ModuleList()
        degrees = [1, 1, 2, 2, 2, 2, 4, 32] if alternative_degrees else [1, 1, 2, 2, 2, 2, 2, 64]
        layers_size = [4, 2, 1, 1, 1, 1, 1]
        features = [96, 96, 256, 256, 256, 128, 128, 128, 3]

        self.truncation = 0.7 if truncate_style else None

        nodes = 1
        num_layers = 7

        # Switcher AdaIN function
        start_index = 1 if after else 0

        for depth in range(num_layers):

            # Mapping
            self.mapping.append(
                MapBlock(features[start_index:], nodes, depth, degrees[start_index:], layers_size, mapping_branching,
                         device))

            # Synthesis
            pt_nodes = nodes if after else nodes * degrees[depth]
            if depth == num_layers - 1:
                self.synthesis.append(
                    PTBlock(features[1:], pt_nodes, depth, degrees[1:], True, False, after))
            else:
                self.synthesis.append(
                    PTBlock(features[1:], pt_nodes, depth, degrees[1:], True, True, after))

            nodes *= degrees[start_index + depth]

    def forward(self, z: torch.Tensor, x: list[torch.Tensor]) -> torch.Tensor:
        for mapping, synthesis in zip(self.mapping, self.synthesis):
            z = mapping(z)

            if self.truncation:
                avg = torch.mean(z, dim=0)
                z_tilde = avg + self.truncation * (z - avg)
            else:
                z_tilde = z

            x = synthesis(x, z_tilde)
        return x[-1]


class TreeGenerator(torch.nn.Module):

    def __init__(self, alternative_degrees: bool):
        super(TreeGenerator, self).__init__()
        degrees = [2, 2, 2, 2, 4, 32] if alternative_degrees else [2, 2, 2, 2, 2, 64]
        features = [96, 64, 64, 64, 64, 64, 3]

        self.synthesis = torch.nn.ModuleList()

        nodes = 1
        num_layers = 6

        for depth in range(num_layers):

            # TreeGAN layers
            if depth == num_layers - 1:
                self.synthesis.append(TreeGCN(features, nodes, depth, degrees, branching=True, activation=False))
            else:
                self.synthesis.append(TreeGCN(features, nodes, depth, degrees, branching=True, activation=True))
            nodes *= degrees[depth]

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        for synthesis in self.synthesis:
            x = synthesis.forward(x)
        return x[-1]
