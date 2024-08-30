import os.path
import shutil

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import metric
from data import CloudTensorDataset
from discriminator import DynamicEdgeDiscriminator
from generator import StyleTreeGenerator
from loss import WassersteinGAN

torch.manual_seed(42)
torch.set_default_dtype(torch.float32)
dir_name = os.path.dirname(__file__)

model_name = 'Surface-Dynamic-3-StyleGAN-64-continue'
model_to_train = 'model/Surface-Dynamic-3-StyleGAN-64'

if __name__ == '__main__':

    # Setting
    device = 'cuda'
    batch_size = 10
    epochs = 300
    iteration = 3

    ada_in_after = False
    mapping_branching = False
    truncate_style = False
    alternative_degrees = False

    # Definition of GAN
    gen = StyleTreeGenerator(ada_in_after, mapping_branching, truncate_style, alternative_degrees, device).to(device)
    dis = DynamicEdgeDiscriminator(batch_size, device).to(device)

    #  Optimizer
    gen_optim = torch.optim.Adam(gen.parameters(), lr=0.0001, betas=(0, 0.99))
    dis_optim = torch.optim.Adam(dis.parameters(), lr=0.0001, betas=(0, 0.99))

    if model_to_train is None:
        os.makedirs(os.path.join(dir_name, model_name))
        os.makedirs(os.path.join(dir_name, model_name, 'loss'))
        os.makedirs(os.path.join(dir_name, model_name, 'jsd'))
        os.makedirs(os.path.join(dir_name, model_name, 'mmd'))
        os.makedirs(os.path.join(dir_name, model_name, 'generator'))

        starting_epoch = 1

        # Metrics Log
        log = open(os.path.join(dir_name, model_name, 'log.csv'), "w")
        log.write("Loss, MMD, JSD\n")
        log.flush()

    else:
        shutil.copytree(os.path.join(dir_name, model_to_train), os.path.join(dir_name, model_name))

        state = torch.load(os.path.join(dir_name, model_name, 'checkpoint.pt'))
        gen.load_state_dict(state['generator'])
        dis.load_state_dict(state['discriminator'])
        gen_optim.load_state_dict(state['generator_optimizer'])
        dis_optim.load_state_dict(state['discriminator_optimizer'])

        starting_epoch = state['epoch'] + 1

        log = open(os.path.join(dir_name, model_name, 'log.csv'), "a")

    # Dataset and DataLoader
    # We assume dataset store in cpu, maybe we can consider to store in on gpu ram
    dataset = CloudTensorDataset(os.path.join(dir_name, 'dataset', 'surface-no-rotated.pt'))

    if device == 'cuda':
        loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True,
                                             pin_memory_device=device)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Loss
    loss = WassersteinGAN(batch_size, device)
    gen.train()
    dis.train()

    for epoch in tqdm(range(starting_epoch, epochs + 1)):

        epoch_loss = 0

        for batch in loader:

            batch = batch.to(device)

            for _ in range(iteration):
                # Discriminator
                dis_optim.zero_grad()

                fake = gen.forward(torch.randn((batch_size, 1, 96), device=device),
                                   [torch.randn((batch_size, 1, 96), device=device)])

                batch_result = dis.forward(batch)
                fake_result = dis.forward(fake)
                loss_dis = loss.discriminator(dis, batch, fake, batch_result, fake_result)
                del fake

                epoch_loss += abs(loss_dis.item())

                loss_dis.backward()

                dis_optim.step()
            del batch

            # Generator
            gen_optim.zero_grad()

            fake = gen.forward(torch.randn((batch_size, 1, 96), device=device),
                               [torch.randn((batch_size, 1, 96), device=device)])
            fake_result = dis.forward(fake)
            del fake

            loss_gen = loss.generator(fake_result)
            loss_gen.backward()

            gen_optim.step()

        # Validation
        fakes = torch.Tensor()

        noises = torch.load(os.path.join(dir_name, 'dataset', 'validation', 'noise.pt')).reshape(-1, 10, 1, 96)
        styles = torch.load(os.path.join(dir_name, 'dataset', 'validation', 'style.pt')).reshape(-1, 10, 1, 96)

        for noise, style in zip(noises, styles):
            fakes = torch.cat((fakes, gen.forward(style.to(device), [noise.to(device)]).detach().cpu()), dim=0)
        del noises, styles

        real = next(iter(DataLoader(dataset, len(dataset), num_workers=2)))

        epoch_loss = epoch_loss / (iteration * (len(dataset) // batch_size))
        mmd, _ = metric.mmd_and_coverage(real, fakes, 100, 'cuda', False)
        jsd = metric.jensen_shannon_entropy(real.numpy(), fakes.numpy(), False)
        del fakes, real

        log.write(f"{epoch_loss:.8f}, {mmd:.8f}, {jsd:.8f}\n")
        log.flush()

        # Save the best model each 100 epochs
        if epoch % 100 == 0:
            torch.save(gen.state_dict(), os.path.join(dir_name, model_name, f'generator-{epoch}.pt'))

    # Checkpoint to resume training
    torch.save({'epoch': epochs,
                'generator': gen.state_dict(),
                'discriminator': dis.state_dict(),
                'generator_optimizer': gen_optim.state_dict(),
                'discriminator_optimizer': dis_optim.state_dict()},
               os.path.join(dir_name, model_name, 'checkpoint.pt'))
