import torch
from torch import Tensor, autograd


class WassersteinGAN:

    def __init__(self, batch_size: int, device: str):
        self.gradient_penalty = self.GradientPenalty(batch_size, device=device)

    class GradientPenalty:

        def __init__(self, batch_size: int, device: str, lambda_gp: float = 10, gamma: float = 1):
            self.batch_size = batch_size
            self.lamda_gp = lambda_gp
            self.gamma = gamma
            self.device = device

        def __call__(self, discriminator: torch.nn.Module, x_real: Tensor, x_fake: Tensor) -> Tensor:
            alpha = torch.rand(self.batch_size, 1, 1, device=self.device, requires_grad=True)
            x_fake = x_fake
            x_real = x_real
            x = x_real + alpha * (x_fake - x_real)

            y = discriminator.forward(x)

            gradients = (autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones(y.size()).to(self.device),
                                       create_graph=True, retain_graph=True, only_inputs=True)[0].contiguous()
                         .view(self.batch_size, -1))

            gradient_penalty = (((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2).mean() * self.lamda_gp
            return gradient_penalty

    def discriminator(self, discriminator: torch.nn.Module, x_real: Tensor, x_fake: Tensor, y_real: Tensor,
                      y_fake: Tensor) -> Tensor:
        dis_loss = -y_real.mean() + y_fake.mean()
        loss_gp = self.gradient_penalty(discriminator, x_real, x_fake)
        dis_loss_gp = dis_loss + loss_gp
        return dis_loss_gp

    @staticmethod
    def generator(y_fake: Tensor) -> Tensor:
        return -y_fake.mean()
