import os.path
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import metric
from data import CloudTensorDataset
from discriminator import Discriminator
from generator import Generator
from loss import WassersteinGAN

if __name__ == '__main__':

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)
    dir_name = os.path.dirname(__file__)

    device = 'cuda'
    ada_in_after = False

    # Definition of GAN
    gen = Generator(ada_in_after, device).to(device)
    dis = Discriminator().to(device)

    #  Optimizer
    gen_optim = torch.optim.Adam(gen.parameters(), lr=0.0001, betas=(0, 0.99))
    dis_optim = torch.optim.Adam(dis.parameters(), lr=0.0001, betas=(0, 0.99))

    # Dataset and DataLoader
    # We assume dataset store in cpu, maybe we can consider to store in on gpu ram
    batch_size = 10
    dataset = CloudTensorDataset(os.path.join(dir_name, 'dataset', 'chair.pt'))

    if device == 'cuda':
        loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True,
                                             pin_memory_device=device)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Loss
    loss = WassersteinGAN(batch_size, device)

    # Setting training
    epochs = 1000
    iteration = 5

    # Metrics Log
    log = open(os.path.join(dir_name, 'model', 'log.csv'), "w")
    log.write("JSD\n")
    log.flush()

    gen.train()
    dis.train()

    # To save metrics
    best_jsd = 1
    best_gen_epoch = 0
    best_gen_state = None

    for epoch in tqdm(range(1, epochs + 1)):

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
                loss_dis.backward()

                dis_optim.step()

            # Generator
            gen_optim.zero_grad()

            fake = gen.forward(torch.randn((batch_size, 1, 96), device=device),
                               [torch.randn((batch_size, 1, 96), device=device)])
            fake_result = dis.forward(fake)

            loss_gen = loss.generator(fake_result)
            loss_gen.backward()

            gen_optim.step()

        # Metric
        fakes = []
        for i in range(len(dataset) // batch_size):
            fakes.append(gen.forward(torch.randn((batch_size, 1, 96), device=device),
                                     [torch.randn((batch_size, 1, 96), device=device)]).detach().cpu())

        fakes = torch.cat(fakes, dim=0)
        real = next(iter(DataLoader(dataset, len(dataset), num_workers=2)))
        jsd = metric.jensen_shannon_entropy(real.numpy(), fakes.numpy(), False)
        log.write(f"{jsd:.4f}\n")
        log.flush()

        # Save the best model each 50 epochs
        if jsd < best_jsd:
            best_jsd = jsd
            best_gen_epoch = epoch
            best_gen_state = deepcopy(gen.state_dict())

        if epoch % 100 == 0:
            torch.save(best_gen_state, os.path.join(dir_name, 'model', f'generator-{best_gen_epoch}.pt'))

    # Checkpoint to resume training
    torch.save(best_gen_state, os.path.join(dir_name, 'model', f'generator-{best_gen_epoch}.pt'))

    torch.save({'epoch': epochs,
                'generator': gen.state_dict(),
                'discriminator': dis.state_dict(),
                'generator_optimizer': gen_optim.state_dict(),
                'discriminator_optimizer': dis_optim.state_dict()},
               os.path.join(dir_name, 'model', 'checkpoint.pt'))
