import math
import os.path
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import metric
from data import CloudTensorDataset
from discriminator import DynamicEdgeDiscriminator
from generator import StyleTreeGenerator
from loss import WassersteinGAN

if __name__ == '__main__':

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)
    dir_name = os.path.dirname(__file__)
    output_dir = 'model'

    os.makedirs(os.path.join(dir_name, output_dir))
    os.makedirs(os.path.join(dir_name, output_dir, 'loss'))
    os.makedirs(os.path.join(dir_name, output_dir, 'jsd'))
    os.makedirs(os.path.join(dir_name, output_dir, 'mmd'))

    # Setting
    device = 'cuda'
    batch_size = 10
    epochs = 100
    iteration = 5

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

    # Metrics Log
    log = open(os.path.join(dir_name, output_dir, 'log.csv'), "w")
    log.write("Loss, MMD, JSD\n")
    log.flush()

    gen.train()
    dis.train()

    # To save metrics
    # loss
    best_loss = math.inf
    best_epoch_loss = 0
    best_state_loss = None

    # jsd
    best_jsd = math.inf
    best_epoch_jsd = 0
    best_state_jsd = None

    # mmd
    best_mmd = math.inf
    best_epoch_mmd = 0
    best_state_mmd = None

    for epoch in tqdm(range(1, epochs + 1)):

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

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch_loss = epoch
            best_state_loss = deepcopy(gen.state_dict())

        if mmd < best_mmd:
            best_mmd = mmd
            best_epoch_mmd = epoch
            best_state_mmd = deepcopy(gen.state_dict())

        if jsd < best_jsd:
            best_jsd = jsd
            best_epoch_jsd = epoch
            best_state_jsd = deepcopy(gen.state_dict())

        # Save the best model each 100 epochs
        if epoch % 100 == 0:
            torch.save(best_state_loss, os.path.join(dir_name, output_dir, 'loss', f'generator-{best_epoch_loss}.pt'))
            torch.save(best_state_mmd, os.path.join(dir_name, output_dir, 'mmd', f'generator-{best_epoch_mmd}.pt'))
            torch.save(best_state_jsd, os.path.join(dir_name, output_dir, 'jsd', f'generator-{best_epoch_jsd}.pt'))

    # Checkpoint to resume training
    torch.save({'epoch': epochs,
                'generator': gen.state_dict(),
                'discriminator': dis.state_dict(),
                'generator_optimizer': gen_optim.state_dict(),
                'discriminator_optimizer': dis_optim.state_dict()},
               os.path.join(dir_name, output_dir, 'checkpoint.pt'))
