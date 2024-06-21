import os.path

import torch
from tqdm import tqdm

from generator import StyleTreeGenerator
from metric import mmd_and_coverage, jensen_shannon_entropy

if __name__ == '__main__':

    # Default Setting
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)
    dir_name = os.path.dirname(__file__)

    # Training-Name
    name = 'StyleTreeGAN'
    degree = 64
    epoch = 971
    metric_name = "mmd"

    # Network setting
    ada_in_after = False
    mapping_branching = False
    truncate_style = False

    model = StyleTreeGenerator(ada_in_after, mapping_branching, truncate_style, degree == 32, "cpu")
    state = torch.load(os.path.join(dir_name, 'model', f'Surface{name}-{degree}', metric_name, f'generator-{epoch}.pt'))
    model.load_state_dict(state)

    # Generation
    fake = torch.Tensor()

    noise = torch.load(os.path.join(dir_name, 'dataset', 'test', 'noise.pt')).reshape(-1, 10, 1, 96)
    style = torch.load(os.path.join(dir_name, 'dataset', 'test', 'style.pt')).reshape(-1, 10, 1, 96)

    for batch, s in tqdm(zip(noise, style), total=len(noise)):
        fake = torch.cat((fake, model.forward(s, [batch]).detach()), 0)

    del model, state, noise

    # Metrics
    batch_size = 100

    real = torch.load(os.path.join(dir_name, 'dataset', 'surface-no-rotated.pt'))
    real = real.reshape(-1, batch_size, 2048, 3)
    fake = fake.reshape(-1, batch_size, 2048, 3)

    cov, mmd, jsd = 0, 0, 0

    for f, r in zip(fake, real):
        x = mmd_and_coverage(r, f, 10, "cuda", True)
        mmd = mmd + x[0]
        cov = cov + x[1]

        jsd += jensen_shannon_entropy(r.numpy(), f.numpy(), False)

    print(cov / (real.size(0)), mmd / real.size(0), jsd / real.size(0))
