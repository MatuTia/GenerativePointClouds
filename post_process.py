import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt
from open3d.cuda.pybind.t.geometry import PointCloud
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from generator import StyleTreeGenerator


class PlotUtils:

    def __init__(self, title: str):
        self.ax = plt.figure().add_subplot(projection='3d')
        self.ax.view_init(elev=0, azim=0, roll=0)
        self.ax.set_title(title)
        plt.axis('off')

    def scatter_plot(self, x, y, z):
        self.ax.scatter3D(x, y, z)

    def perimeter_plot(self, x, y, z):
        self.ax.plot3D(x, y, z)

    def plane_plot(self, x, y, z):
        self.ax.plot_surface(x, y, z)

    @staticmethod
    def show_plot():
        plt.show()


def compute_boundary_interior_points(cloud: np.ndarray) -> [np.ndarray, np.ndarray]:

    cloud = PointCloud(cloud)
    cloud.estimate_normals(radius=0.2)
    _, mask = cloud.compute_boundary_points(radius=0.2, max_nn=100)

    cloud = cloud.point.positions.numpy()
    mask = mask.numpy()

    return cloud[mask], cloud[~mask]


def compute_perimeter(boundary):
    # Choose the first element of perimeter
    start_idx = np.argmin(np.sum(boundary, axis=1))
    point = boundary[start_idx]
    start = [point]

    # Star perimeter
    boundary = np.concatenate((boundary[:start_idx], boundary[start_idx + 1:]))
    half, quarter = boundary.shape[0] // 2, boundary.shape[0] // 4

    perimeter = [point]

    # Define perimeter
    while len(boundary) > 0:
        distance = cdist([point], boundary)
        if ((len(perimeter) > half and np.min(distance) > .2) or
                (len(boundary) < quarter and cdist([point], start) < np.min(distance))):
            break

        idx = np.argmin(distance)

        point = boundary[idx]
        perimeter.append(point)
        boundary = np.concatenate((boundary[:idx], boundary[idx + 1:]))

    perimeter = np.asarray(perimeter)
    closest = np.argmin(cdist(perimeter[:10], perimeter[-1:]))
    return perimeter[closest:]


def local_linear_regression_slopes(perimeter, num_neighbors):
    def linear_regression(x, y):
        regression = LinearRegression()
        regression.fit(x.reshape(-1, 1), y)
        return regression

    # Since we know the relation between variables
    projection = perimeter[:, :2]

    slopes = []
    for i in range(len(projection)):
        index = np.array(list(range(i - num_neighbors - 1, i - 1)) + list(range(i + 1, i + num_neighbors + 1)))
        index = np.mod(index, len(projection))
        local = projection[index].T
        slopes.append([linear_regression(local[0], local[1]).coef_.item(),
                       linear_regression(local[1], local[0]).coef_.item()])

    return np.asarray(slopes)


def compute_candidate(slopes):
    slopes_y, slopes_x = slopes[:, 0], slopes[:, 1]

    candidate = []
    for i in range(slopes_y.shape[0]):
        y_old, x_old = slopes_y[(i - 1) % slopes_y.shape[0]], slopes_x[(i - 1) % slopes_x.shape[0]]
        y_new, x_new = slopes_y[i], slopes_x[i]
        if y_old * y_new <= 0 or np.argmax((x_old, y_old)) != np.argmax((x_new, y_new)):
            near_zero = (i - 1) % slopes_y.shape[0] if np.abs(slopes_y[i]) - np.abs(slopes_y[i - 1]) >= 0 else i
            candidate.append(near_zero)

    candidate = np.unique(np.array(candidate))
    return candidate


def find_corners(candidate, perimeter):
    corners = np.zeros(4, dtype=int)

    point = perimeter[candidate]
    distance = cdist(point, point)
    diagonal = np.argmax(distance)
    index1 = diagonal // len(candidate)
    index2 = diagonal % len(candidate)
    corners[0] = candidate[index1]
    corners[2] = candidate[index2]

    distance = np.sum(distance[[index1, index2]], axis=0)
    corners[1] = _find_corner(index1, index2, candidate, perimeter, distance)
    corners[3] = _find_corner(index2, index1, candidate, perimeter, distance)

    return np.sort(corners)


def _find_corner(index1, index2, candidate, perimeter, distance):
    length = perimeter.shape[0]

    corner1 = candidate[index1]
    corner2 = candidate[index2]

    if corner1 > corner2:
        candidate = np.where(candidate <= corner2, candidate + length, candidate)
        corner2 += length

    points = np.zeros(3)

    optimal = corner1 + (corner2 - corner1) // 2
    points[0] = candidate[np.argmin(np.abs(candidate - optimal))] % length

    side = length // 4

    optimal = corner1 + side
    points[1] = candidate[np.argmin(np.abs(candidate - optimal))] % length

    optimal = corner2 - side
    points[2] = candidate[np.argmin(np.abs(candidate - optimal))] % length

    points = np.unique(points)

    if len(points) == 1:
        return points[0]

    index = np.searchsorted(candidate, points)
    return points[np.argmax(distance[index])]


def point_mask(perimeter, interior, corners):
    mask = np.zeros(perimeter.shape[0])
    for i in range(1, 4):
        mask[corners[i - 1]:corners[i]] = i
    mask += 1

    mask = np.concatenate((mask[corners[-1]:], mask[:corners[-1]]))
    mask = np.concatenate((mask, np.zeros(interior.shape[0])))

    perimeter = np.concatenate((perimeter[corners[-1]:], perimeter[:corners[-1]]))
    point = np.concatenate((perimeter, interior))

    point = np.concatenate((point, mask.reshape(-1, 1)), axis=1)

    return point, mask


def normalization(clouds: np.ndarray) -> np.ndarray:
    def _normalization(cloud):
        index = cloud[:, -1:]
        normalized = (cloud[:, :3] - left) / (right - left)
        return np.concatenate([normalized, index], axis=1)

    concatenated = np.concatenate(clouds, axis=0)[:, :3]
    left, right = np.min(concatenated), np.max(concatenated)

    clouds = list(tqdm(map(_normalization, clouds), total=len(clouds)))
    clouds = np.asarray(clouds, dtype=object)

    return clouds


def rotation(clouds: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    def _rotation(cloud):
        angle = np.random.rand() * np.pi
        axis = np.zeros(3)
        axis[np.random.randint(3)] = 1
        index = cloud[:, -1:]
        rotated = Rotation.from_rotvec(angle * axis).apply(cloud[:, :3])
        return np.concatenate([rotated, index], axis=1)

    clouds = list(tqdm(map(_rotation, clouds), total=len(clouds)))
    clouds = np.asarray(clouds, dtype=object)

    return clouds


def post_process(cloud: np.ndarray, plotting: bool) -> np.ndarray:
    boundary, interior = compute_boundary_interior_points(cloud)

    perimeter = compute_perimeter(boundary)

    slopes = local_linear_regression_slopes(perimeter, num_neighbors=8)

    candidate = compute_candidate(slopes)

    corners = find_corners(candidate, perimeter) if len(candidate) > 4 else candidate

    if np.unique(corners).shape[0] != 4:
        size = len(slopes) // 4
        corners = [i * size for i in range(4)]

    point, mask = point_mask(perimeter, interior, corners)

    if plotting:
        perimeter = point[np.where(mask != 0, True, False)]
        plot = PlotUtils("")

        plot.perimeter_plot(perimeter[:, 2], perimeter[:, 0], perimeter[:, 1])

        point = perimeter[np.where(mask == 1)].T
        plot.scatter_plot(point[2], point[0], point[1])

        for i, color in enumerate(['green', 'red', 'blue']):
            point = perimeter[np.where(mask == i + 2)].T
            plot.scatter_plot(point[2], point[0], point[1])

        plot.scatter_plot(interior[:, 2], interior[:, 0], interior[:, 1])
        plot.show_plot()

    return point


def boundary_parametrization(clouds):
    def uniform_parametrization(p):
        n = p.shape[0]
        return torch.arange(n) / (n - 1)

    def _boundary_parametrization(c):
        index = c[:, 3]
        index = np.where(index == 0, False, True)
        boundary, interior = c[index], c[~index]

        index = boundary[:, 3]

        axis = np.argwhere(index == 1)

        axis = boundary[:(axis[-1] + 1).item()]
        partition = np.stack((uniform_parametrization(axis)[:-1], np.zeros(len(axis) - 1)), axis=-1)

        axis = np.argwhere(index == 2)
        axis = boundary[axis[0].item():(axis[-1] + 1).item()]
        tmp = np.stack((np.ones(len(axis) - 1), uniform_parametrization(axis)[:-1]), axis=-1)
        partition = np.concatenate((partition, tmp))

        axis = np.argwhere(index == 3)
        axis = boundary[axis[0].item():(axis[-1] + 1).item()]
        tmp = np.stack((uniform_parametrization(axis).flip(0)[:-1], np.ones(len(axis) - 1)), axis=-1)
        partition = np.concatenate((partition, tmp))

        axis = np.argwhere(index == 4)
        axis = np.concatenate((boundary[axis[0].item():], boundary[:1]))
        tmp = np.stack((np.zeros(len(axis) - 1), uniform_parametrization(axis).flip(0)[:-1]), axis=-1)
        partition = np.concatenate((partition, tmp))

        return partition

    boundaries = []
    interiors = []

    for cloud in tqdm(clouds):
        parameterization = _boundary_parametrization(cloud)
        boundaries.append(np.concatenate((cloud[:len(parameterization), :3], parameterization), axis=1))
        interiors.append(cloud[len(parameterization):, :3])

    return np.asarray(boundaries, dtype=object), np.asarray(interiors, dtype=object)


def main():
    # Generator
    if generate_clouds:
        model = StyleTreeGenerator(ada_in_after, mapping_branching, truncate_style, alternative_degrees, device)
        model = model.to(device)
        state = torch.load(model_name, device)
        model.load_state_dict(state)

        clouds = np.empty((num_cloud, 2048, 3), dtype=np.float32)
        iteration = num_cloud // batch_size

        for idx, iteration in tqdm(enumerate(range(iteration)), total=iteration):
            style = torch.randn(batch_size, 1, 96, dtype=torch.float32, device=device)
            noise = [torch.randn(batch_size, 1, 96, dtype=torch.float32, device=device)]

            clouds[idx * batch_size:(idx + 1) * batch_size] = model.forward(style, noise).detach().cpu().numpy()

        np.save(os.path.join(output_dir, name_dataset), clouds)

        del model, clouds

    clouds = np.load(os.path.join(output_dir, name_dataset))

    # Extract Perimeter
    output = []
    for cloud in tqdm(clouds):
        output.append(post_process(cloud, False))

    output = np.asarray(output, dtype=object)
    np.save(os.path.join(output_dir, "post-process-clouds.npy"), output)

    del output, clouds

    clouds = np.load(os.path.join(output_dir, "post-process-clouds.npy"), allow_pickle=True)

    # Other transformation
    clouds = rotation(clouds)
    clouds = normalization(clouds)

    # Parametrization
    boundary, interior = boundary_parametrization(clouds)

    np.save(os.path.join(output_dir, 'boundary.npy'), boundary)
    np.save(os.path.join(output_dir, 'interior.npy'), interior)


if __name__ == '__main__':

    # Default Setting
    torch.set_default_dtype(torch.float32)
    torch.random.manual_seed(42)
    device = 'cpu'

    # Directory Name
    dir_name = os.path.dirname(__file__)
    output_dir = os.path.join(dir_name, 'clouds-generated')
    name_dataset = "generated-clouds.npy"

    # Generation
    generate_clouds = True

    if generate_clouds:
        # Model Setting
        ada_in_after = False
        mapping_branching = False
        truncate_style = False
        alternative_degrees = False
        name = 'Surface-Dynamic-3-StyleGAN-64'

        model_name = os.path.join(dir_name, 'model', name, 'generator', 'generator-500.pt')

        # Generation Setting
        num_cloud = 100
        batch_size = 10

        os.makedirs(output_dir, exist_ok=True)

    main()
