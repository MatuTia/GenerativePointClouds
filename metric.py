import numpy as np
import torch
from pytorch3d.loss import chamfer
from scipy.spatial import distance
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# JSD
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[np.linalg.norm(grid, axis=1) <= 0.5]

    return grid, spacing


def entropy_of_occupancy_grid(clouds, grid_resolution, in_sphere=False):
    """Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        clouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    # epsilon = 10e-4
    # bound = 0.5 + epsilon
    # if abs(np.max(clouds)) > bound or abs(np.min(clouds)) > bound:
    #     warnings.warn('Point-clouds are not in unit cube.')
    #
    # if in_sphere and np.max(np.sqrt(np.sum(clouds ** 2, axis=2))) > bound:
    #     warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_r_vars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in clouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_r_vars[i] += 1

    acc_entropy = 0.0
    n = float(len(clouds))
    for g in grid_bernoulli_r_vars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_entropy(real_cloud, fake_clouds, in_sphere=False):
    _, pk = entropy_of_occupancy_grid(real_cloud, 28, in_sphere)
    _, qk = entropy_of_occupancy_grid(fake_clouds, 28, in_sphere)

    return distance.jensenshannon(pk, qk, base=2) ** 2


# CD
def scheduler_chamfer_distance(x, y, batch_size, device, verbose):
    assert x.size(0) % batch_size == y.size(0) % batch_size == 0

    result = torch.zeros(x.size(0), x.size(0))

    iteration = range(x.size(0) // batch_size)
    iteration = tqdm(iteration) if verbose else iteration

    for i in iteration:
        x_batch = x[i * batch_size: (i + 1) * batch_size].to(device)

        for j in range(y.size(0) // batch_size):
            y_batch = y[j * batch_size: (j + 1) * batch_size].to(device)

            out = chamfer_distance(x_batch, y_batch).cpu()

            result[i * batch_size: (i + 1) * batch_size, j * batch_size: (j + 1) * batch_size] = out

    return result


def chamfer_distance(x, y):
    assert x.size() == y.size()
    batch_size = x.size(0)
    x = x.repeat(1, x.size(0), 1).view(-1, x.size(1), x.size(2)).cuda()
    y = y.repeat(y.size(0), 1, 1).cuda()
    # The average get the result independent of the number of points
    # We use it, although they define the function with sum
    result = chamfer.chamfer_distance(x, y, batch_reduction=None, point_reduction="mean")[0]
    return result.view(batch_size, batch_size)


def mmd_and_coverage(real, fake, batch_size, device, verbose):
    matrix = scheduler_chamfer_distance(real, fake, batch_size, device, verbose)

    # From real to fake
    mmd, _ = matrix.min(dim=0)
    # From fake to real
    _, cov = matrix.min(dim=1)

    return mmd.mean().item(), cov.unique().size(0)
