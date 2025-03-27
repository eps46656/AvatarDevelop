import pathlib

import numpy as np

import camera_utils
import rendering_utils
import utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


def main1():
    H = 720
    W = 1280

    origin = np.array([0, 0, 0], dtype=float)
    x_axis = np.array([1, 0, 0], dtype=float)
    y_axis = np.array([0, 1, 0], dtype=float)
    z_axis = np.array([0, 0, 1], dtype=float)

    raduis = 10
    theta = 45 * utils.DEG
    phi = (180 + 270) / 2 * utils.DEG

    proj_mat = camera_utils.make_proj_mat(
        img_shape=(H, W),
        origin=np.array(utils.Sph2Cart(raduis, theta, phi)),
        # origin=np.array([4, 5, 6, 7]),
        aim=origin,
        quasi_u_dir=z_axis,
        diag_fov=45 * utils.DEG,
    )[:3, :]

    print(f"{proj_mat}")

    point_a = np.array([[0], [0], [0], [1]], dtype=float)
    point_x_pos = np.array([[1], [0], [0], [1]], dtype=float)
    point_x_neg = np.array([[-1], [0], [0], [1]], dtype=float)
    point_y_pos = np.array([[0], [1], [0], [1]], dtype=float)
    point_y_neg = np.array([[0], [-1], [0], [1]], dtype=float)
    point_z_pos = np.array([[0], [0], [1], [1]], dtype=float)
    point_z_neg = np.array([[0], [0], [-1], [1]], dtype=float)

    points = list()
    points.append(point_x_pos)
    points.append(point_x_neg)
    points.append(point_y_pos)
    points.append(point_y_neg)
    points.append(point_z_pos)
    points.append(point_z_neg)

    img = np.zeros((H, W, 3), dtype=np.uint8)

    for point in points:
        img_point = camera_utils.HomographyMul(
            proj_mat, point).flatten()

        print(f"{img_point=}")

        for (x, y), (u, v) in rendering_utils.RasterizeEllipseWithAxis(
                img_point[:2], np.array([10, 0]), np.array([0, 10]), (0, H), (0, W)):

            img[x, y, :] = (255, 0, 0)

    utils.write_image(DIR / "test.png", img)


def main2():
    for _ in range(1024):
        x, y, z = np.random.rand(3) * 10

        radius, theta, phi = utils.Cart2Sph(x, y, z)

        re_x, re_y, re_z = utils.Sph2Cart(radius, theta, phi)

        err = np.linalg.norm(
            np.array([x, y, z]) - np.array([re_x, re_y, re_z]))

        assert err <= 1e-5

        print(f"{err=}")


if __name__ == "__main__":
    main1()
