import math

import numpy as np
from typeguard import typechecked
import torch

import utils


@typechecked
def ArrangeAxesStr(axes: str):
    assert len(axes) == 3

    axes = axes.lower()

    assert sum(c in axes for c in "lr") == 1
    assert sum(c in axes for c in "ud") == 1
    assert sum(c in axes for c in "fb") == 1

    return axes


xyz_to_urf_table = {
    "u": (0, +1), "d": (0, -1),
    "l": (1, -1), "r": (1, +1),
    "f": (2, +1), "b": (2, -1),
}


@typechecked
def MakeXYZToURFMat(axes: str):
    axes = ArrangeAxesStr(axes)

    ret = torch.zeros(3, 3, dtype=torch.float)

    for k in range(3):
        p = xyz_to_urf_table[axes[k]]
        ret[p[0], k] = p[1]

    return ret


@typechecked
def MakeProjMatWithURF(
    *,
    img_shape: tuple[int, int],
    origin: torch.Tensor,
    u_vec: torch.Tensor,
    r_vec: torch.Tensor,
    f_vec: torch.Tensor,
):
    assert 0 < img_shape[0]
    assert 0 < img_shape[1]

    origin = origin.flatten()
    u_vec = u_vec.flatten()
    r_vec = r_vec.flatten()
    f_vec = f_vec.flatten()

    assert origin.shape[0] == 3
    assert u_vec.shape[0] == 3
    assert r_vec.shape[0] == 3
    assert f_vec.shape[0] == 3

    assert abs(utils.GetAngle(f_vec, u_vec) - 90 * utils.DEG) < utils.EPS
    assert abs(utils.GetAngle(f_vec, r_vec) - 90 * utils.DEG) < utils.EPS

    h_vec = u_vec * (-2 / (img_shape[0] - 1))
    w_vec = r_vec * (2 / (img_shape[1] - 1))
    ul_vec = f_vec + u_vec - r_vec

    return torch.inverse(torch.tensor([
        [h_vec[0], w_vec[0], ul_vec[0], origin[0]],
        [h_vec[1], w_vec[1], ul_vec[1], origin[1]],
        [h_vec[2], w_vec[2], ul_vec[2], origin[2]],
        [0, 0, 0, 1],
    ], dtype=torch.float))


@typechecked
def MakeViewMatWithURF(
    *,
    origin: torch.Tensor,  # [..., 3]
    u_vec: torch.Tensor,  # [..., 3]
    r_vec: torch.Tensor,  # [..., 3]
    f_vec: torch.Tensor,  # [..., 3]
    view_axes: str,
    dtype: torch.dtype,
    device: torch.device,
):
    view_axes = ArrangeAxesStr(view_axes)

    utils.CheckShapes(
        origin, (..., 3),
        u_vec, (..., 3),
        r_vec, (..., 3),
        f_vec, (..., 3),
    )

    def GetVec(axis):
        match axis:
            case "l": return -r_vec
            case "r": return r_vec
            case "u": return u_vec
            case "d": return -u_vec
            case "f": return f_vec
            case "b": return -f_vec

    x_vec = GetVec(view_axes[0])
    y_vec = GetVec(view_axes[1])
    z_vec = GetVec(view_axes[2])

    batch_dims = list(utils.GetCommonShape([
        x_vec.shape[:-1],
        y_vec.shape[:-1],
        z_vec.shape[:-1],
    ]))

    ret = torch.empty(batch_dims + [4, 4], dtype=dtype, device=device)

    ret[..., 0, 0] = x_vec[..., 0]
    ret[..., 0, 1] = y_vec[..., 0]
    ret[..., 0, 2] = z_vec[..., 0]
    ret[..., 0, 3] = origin[..., 0]

    ret[..., 1, 0] = x_vec[..., 1]
    ret[..., 1, 1] = y_vec[..., 1]
    ret[..., 1, 2] = z_vec[..., 1]
    ret[..., 1, 3] = origin[..., 1]

    ret[..., 2, 0] = x_vec[..., 2]
    ret[..., 2, 1] = y_vec[..., 2]
    ret[..., 2, 2] = z_vec[..., 2]
    ret[..., 2, 3] = origin[..., 2]

    ret[..., 3, :3] = 0
    ret[..., 3, 3] = 1

    return torch.inverse(ret)


@typechecked
def MakeProjMat(
    *,
    img_shape: tuple[int, int],
    origin: torch.Tensor,
    aim: torch.Tensor,
    quasi_u_dir: torch.Tensor,
    diag_fov: float,
):
    assert 0 < img_shape[0]
    assert 0 < img_shape[1]

    assert origin.shape.numel() == 3
    assert aim.shape.numel() == 3
    assert quasi_u_dir.shape.numel() == 3

    origin = origin.flatten()
    aim = aim.flatten()
    quasi_u_dir = quasi_u_dir.flatten()

    f_dir = aim - origin
    f_dir_len = torch.norm(f_dir)

    assert utils.EPS < f_dir_len

    f_vec = f_dir / f_dir_len

    r_dir = torch.cross(f_vec, quasi_u_dir, dim=0)
    r_dir_norm = torch.norm(r_dir)

    assert utils.EPS < r_dir_norm

    u_dir = torch.cross(r_dir, f_vec, dim=0)
    u_dir_norm = torch.norm(u_dir)

    assert utils.EPS < u_dir_norm

    half_diag_len = math.tan(diag_fov / 2)

    img_diag_len = math.sqrt(img_shape[0]**2 + img_shape[1]**2)

    u_vec = u_dir * (img_shape[0] / img_diag_len * half_diag_len / u_dir_norm)
    r_vec = r_dir * (img_shape[1] / img_diag_len * half_diag_len / r_dir_norm)

    assert abs(img_shape[0] / img_shape[1] -
               torch.norm(u_vec) / torch.norm(r_vec)) < utils.EPS

    return MakeProjMatWithURF(
        img_shape=img_shape,
        origin=origin,
        u_vec=u_vec,
        r_vec=r_vec,
        f_vec=f_vec,
    )


@typechecked
def MakeViewMat(
    origin: torch.Tensor,  # [..., 3]
    aim: torch.Tensor,  # [..., 3]
    quasi_u_dir: torch.Tensor,  # [..., 3]
    view_axes: str,
    dtype: torch.dtype,
    device: torch.device,
):
    assert 1 <= origin.dim()
    assert 1 <= aim.dim()
    assert 1 <= quasi_u_dir.dim()

    assert origin.shape[-1] == 3
    assert aim.shape[-1] == 3
    assert quasi_u_dir.shape[-1] == 3

    f_vec = utils.Normalized(aim - origin, -1)
    r_vec = utils.Normalized(torch.cross(f_vec, quasi_u_dir, dim=-1), -1)
    u_vec = torch.cross(r_vec, f_vec, dim=0)

    return MakeViewMatWithURF(
        origin=origin,
        u_vec=u_vec,
        r_vec=r_vec,
        f_vec=f_vec,
        view_axes=view_axes,
        dtype=dtype,
        device=device,
    )


@typechecked
def MakePersProjMat(
    *,
    view_axes: str,
    image_shape: tuple[int, int],
    ndc_axes: str,
    diag_fov: float,
    far: float,
):
    assert 0 < image_shape[0]
    assert 0 < image_shape[1]

    view_axes = ArrangeAxesStr(view_axes)
    ndc_axes = ArrangeAxesStr(ndc_axes)

    assert 0 < diag_fov
    assert diag_fov < 180 * utils.DEG

    view_to_urf = torch.eye(4)
    ndc_to_urf = torch.eye(4)

    view_to_urf[:3, :3] = MakeXYZToURFMat(view_axes)
    ndc_to_urf[:3, :3] = MakeXYZToURFMat(ndc_axes)

    """

    (u_dir * x + r_dir * y + f_dir)

    u_dir = [h*k, 0, 0]
    r_dir = [0, w*k, 0]
    f_dir = [0, 0, far]

    """

    assert 0 < far

    h, w = image_shape

    k = math.sqrt((math.tan(diag_fov / 2) * far)**2 / (h**2 + w**2))

    urf_proj_mat = torch.inverse(torch.tensor([
        [k*h, 0, 0, 0],
        [0, k*w, 0, 0],
        [0, 0, 0, 1],
        [0, 0, far, 0],
    ]))

    return torch.inverse(ndc_to_urf) @ urf_proj_mat @ view_to_urf


@typechecked
def GetFocalLengthByDiagFoV(img_size: tuple[int, int], diag_fov: float):
    assert 0 < diag_fov
    assert diag_fov < 180 * utils.DEG

    return math.sqrt(img_size[0]**2 + img_size[1]**2) / (2 * math.tan(diag_fov / 2))


@typechecked
def MakeImageMat(
    image_shape: tuple[int, int],
):
    pass


def HomographyMul(h: torch.Tensor, p: torch.Tensor):
    # h[N, D]
    # p[D, M]

    assert len(h.shape) == 2

    N, D = h.shape

    assert len(p.shape) == 2
    assert p.shape[0] == D

    q = h @ p
    # [N, M]

    return q / q[-1, :]


def main1():
    origin = np.array([0, 0, 0], dtype=torch.float)
    x_axis = np.array([1, 0, 0], dtype=torch.float)
    y_axis = np.array([0, 1, 0], dtype=torch.float)
    z_axis = np.array([0, 0, 1], dtype=torch.float)

    raduis = 10
    theta = 45 * utils.DEG
    phi = (180 + 270) / 2 * utils.DEG

    proj_mat = MakeProjMat(
        img_shape=(1080, 1920),
        origin=np.array(utils.Sph2Cart(raduis, theta, phi)),
        # origin=np.array([4, 5, 6, 7]),
        aim=origin,
        quasi_u_dir=z_axis,
        diag_fov=45 * utils.DEG,
    )

    print(f"{proj_mat}")

    point_a = np.array([[0], [0], [0], [1]], dtype=torch.float)
    point_x_pos = np.array([[1], [0], [0], [1]], dtype=torch.float)
    point_x_neg = np.array([[-1], [0], [0], [1]], dtype=torch.float)
    point_y_pos = np.array([[0], [1], [0], [1]], dtype=torch.float)
    point_y_neg = np.array([[0], [-1], [0], [1]], dtype=torch.float)
    point_z_pos = np.array([[0], [0], [1], [1]], dtype=torch.float)
    point_z_neg = np.array([[0], [0], [-1], [1]], dtype=torch.float)

    points = list()
    points.append(point_x_pos)
    points.append(point_x_neg)
    points.append(point_y_pos)
    points.append(point_y_neg)
    points.append(point_z_pos)
    points.append(point_z_neg)

    H = 720
    W = 1280

    img = np.zeros((H, W, 3), dtype=np.uint8)

    for point in points:
        img_point = HomographyMul(proj_mat, point_x_pos).flatten()

        img[img_point[0], img_point[1], :] = (255, 0, 0)

    utils.WriteImage(DIR / "test.png", img)


def main2():
    for _ in range(1024):
        x, y, z = np.random.rand(3) * 10

        radius, theta, phi = utils.Cart2Sph(x, y, z)

        re_x, re_y, re_z = utils.Sph2Cart(radius, theta, phi)

        err = np.norm(
            np.array([x, y, z]) - np.array([re_x, re_y, re_z]))

        assert err <= 1e-5

        print(f"{err=}")


if __name__ == "__main__":
    main1()
