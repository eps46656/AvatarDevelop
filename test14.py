import contextlib
import dataclasses
import math
import pathlib

import torch
import torch.nn as nn
from einops import reduce

from . import camera_utils, transform_utils, utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cpu")

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """

    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                  C1 * y * sh[..., 1] +
                  C1 * z * sh[..., 2] -
                  C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh[..., 4] +
                      C2[1] * yz * sh[..., 5] +
                      C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                      C2[3] * xz * sh[..., 7] +
                      C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                          C3[1] * xy * z * sh[..., 10] +
                          C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                          C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                          C3[5] * z * (xx - yy) * sh[..., 14] +
                          C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                              C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                              C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                              C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                              C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                              C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                              C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                              C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                              C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def build_rotation(r):
    print(f"{r=}")

    norm = torch.sqrt(r[:, 0]*r[:, 0] + r[:, 1]*r[:, 1] +
                      r[:, 2]*r[:, 2] + r[:, 3]*r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), dtype=utils.FLOAT, device=DEVICE)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=utils.FLOAT, device=DEVICE)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros(
        (L.shape[0], 6), dtype=utils.FLOAT, device=DEVICE)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance
    # symm = strip_symmetric(actual_covariance)
    # return symm


def build_covariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    # The following models the steps outlined by equations 29
    # and 31 in "EWA Splatting" (Zwicker et al., 2002).
    # Additionally considers aspect / scaling of viewport.
    # Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3, :3]) + viewmatrix[-1:, :3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-
                                      tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-
                                      tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    W = viewmatrix[:3, :3].T  # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0, 2, 1)

    # add low pass filter here according to E.q. 32
    filter = torch.eye(2, 2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter[None]


def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points)  # object space
    points_h = points_o @ viewmatrix @ projmatrix  # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2

    print(f"{viewmatrix=}")
    print(f"{projmatrix=}")

    return p_proj, p_view, in_mask


@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()


@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:, None])
    rect_max = (pix_coord + radii[:, None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


class GaussRenderer(nn.Module):
    """
    A gaussian splatting renderer

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel, camera=camera)
    """

    def __init__(self, active_sh_degree=3, white_bkgd=True, **kwargs):
        super(GaussRenderer, self).__init__()
        self.active_sh_degree = active_sh_degree
        self.debug = False
        self.white_bkgd = white_bkgd

    def build_color(self, means3D, shs, camera):
        rays_o = camera.camera_center
        rays_d = means3D - rays_o
        color = eval_sh(self.active_sh_degree, shs.permute(0, 2, 1), rays_d)
        color = (color + 0.5).clip(min=0.0)
        return color

    def render(self, camera, means2D, cov2d, color, opacity, depths):
        radii = get_radius(cov2d)
        rect = get_rect(means2D, radii, width=camera.image_width,
                        height=camera.image_height)

        pix_coord = torch.stack(torch.meshgrid(
            torch.arange(camera.image_width),
            torch.arange(camera.image_height),
            indexing='xy'
        ), dim=-1).to(DEVICE)

        self.render_color = torch.ones(
            *pix_coord.shape[:2], 3).to(DEVICE)
        self.render_depth = torch.zeros(
            *pix_coord.shape[:2], 1).to(DEVICE)
        self.render_alpha = torch.zeros(
            *pix_coord.shape[:2], 1).to(DEVICE)

        TILE_SIZE = 64
        for h in range(0, camera.image_height, TILE_SIZE):
            for w in range(0, camera.image_width, TILE_SIZE):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(
                    min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(
                    max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
                in_mask = (over_br[0] > over_tl[0]) & (
                    over_br[1] > over_tl[1])  # 3D gaussian in the tile

                if not in_mask.sum() > 0:
                    continue

                P = in_mask.sum()
                tile_coord = pix_coord[h:h +
                                       TILE_SIZE, w:w+TILE_SIZE].flatten(0, -2)
                sorted_depths, index = torch.sort(depths[in_mask])
                sorted_means2D = means2D[in_mask][index]
                sorted_cov2d = cov2d[in_mask][index]  # P 2 2
                sorted_conic = sorted_cov2d.inverse()  # inverse of variance
                sorted_opacity = opacity[in_mask][index]
                sorted_color = color[in_mask][index]
                dx = (tile_coord[:, None, :] -
                      sorted_means2D[None, :])  # B P 2

                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, 0]**2 * sorted_conic[:, 0, 0]
                    + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                    + dx[:, :, 0]*dx[:, :, 1] * sorted_conic[:, 0, 1]
                    + dx[:, :, 0]*dx[:, :, 1] * sorted_conic[:, 1, 0]))

                alpha = (gauss_weight[..., None] *
                         sorted_opacity[None]).clip(max=0.99)  # B P 1
                T = torch.cat([torch.ones_like(alpha[:, :1]),
                              1-alpha[:, :-1]], dim=1).cumprod(dim=1)
                acc_alpha = (alpha * T).sum(dim=1)

                tile_color = (T * alpha * sorted_color[None]).sum(
                    dim=1) + (1-acc_alpha) * (1 if self.white_bkgd else 0)
                tile_depth = (
                    (T * alpha) * sorted_depths[None, :, None]).sum(dim=1)
                self.render_color[h:h+TILE_SIZE, w:w +
                                  TILE_SIZE] = tile_color.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_depth[h:h+TILE_SIZE, w:w +
                                  TILE_SIZE] = tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_alpha[h:h+TILE_SIZE, w:w +
                                  TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)

        return {
            "render": self.render_color,
            "depth": self.render_depth,
            "alpha": self.render_alpha,
            "visiility_filter": radii > 0,
            "radii": radii
        }

    def forward(self, camera, pc, **kwargs):
        means3D = pc.means
        opacity = pc.opacities
        scales = pc.scales
        rotations = pc.rots
        shs = pc.shs

        prof = contextlib.nullcontext

        with prof("projection"):
            mean_ndc, mean_view, in_mask = projection_ndc(means3D,
                                                          viewmatrix=camera.world_view_transform,
                                                          projmatrix=camera.projection_matrix)
            mean_ndc = mean_ndc[in_mask]
            mean_view = mean_view[in_mask]
            depths = mean_view[:, 2]

        with prof("build color"):
            color = self.build_color(means3D=means3D, shs=shs, camera=camera)

        with prof("build cov3d"):
            cov3d = build_covariance_3d(scales, rotations)

        with prof("build cov2d"):
            cov2d = build_covariance_2d(
                mean3d=means3D,
                cov3d=cov3d,
                viewmatrix=camera.world_view_transform,
                fov_x=camera.FoVx,
                fov_y=camera.FoVy,
                focal_x=camera.focal_x,
                focal_y=camera.focal_y)

            mean_coord_x = ((mean_ndc[..., 0] + 1) *
                            camera.image_width - 1.0) * 0.5
            mean_coord_y = ((mean_ndc[..., 1] + 1) *
                            camera.image_height - 1.0) * 0.5
            means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)

        with prof("render"):
            rets = self.render(
                camera=camera,
                means2D=means2D,
                cov2d=cov2d,
                color=color,
                opacity=opacity,
                depths=depths,
            )

        return rets


@dataclasses.dataclass
class Camera:
    camera_center: torch.Tensor  # [3]

    image_height: int
    image_width: int

    FoVx: float
    FoVy: float

    focal_x: float
    focal_y: float

    world_view_transform: torch.Tensor  # [4, 4]
    projection_matrix: torch.Tensor  # [4, 4]


@dataclasses.dataclass
class PC:
    means: torch.Tensor
    opacities: torch.Tensor
    scales: torch.Tensor
    rots: torch.Tensor
    shs: torch.Tensor


class GaussianRendererWrapper(torch.nn.Module):
    torch_splatting_view_dirs = utils.Dir3.FromStr("RUF")
    torch_splatting_ndc_dirs = utils.Dir3.FromStr("RUF")

    torch_splatting_view_transform = \
        transform_utils.ObjectTransform.FromDir3(torch_splatting_view_dirs)
    # camera <-> view

    def __init__(self):
        super(GaussianRendererWrapper, self).__init__()

        self.torch_splatting_gauss_renderer = \
            GaussRenderer(active_sh_degree=1,
                          white_bkgd=True)

    @staticmethod
    def MakeCamera(
        camera_transform: transform_utils.ObjectTransform,
        camera_config: camera_utils.CameraConfig,
    ):
        view_mat = camera_transform.get_trans_to(
            GaussianRendererWrapper.torch_splatting_view_transform)
        # world -> view

        print(f"{camera_transform.trans=}")
        print(f"{camera_transform.inv_trans=}")

        print(f"{GaussianRendererWrapper.torch_splatting_view_transform.trans=}")
        print(f"{GaussianRendererWrapper.torch_splatting_view_transform.inv_trans=}")

        ndc_proj_mat = camera_utils.make_proj_mat_with_config(
            camera_config=camera_config,
            view_coord=GaussianRendererWrapper.torch_splatting_view_dirs,
            proj_config=camera_utils.ProjConfig(
                dirs=GaussianRendererWrapper.torch_splatting_ndc_dirs,

                delta_u=1.0,
                delta_d=1.0,

                delta_l=1.0,
                delta_r=1.0,

                delta_f=1.0,
                delta_b=0.0,
            ),
        )
        # view -> ndc

        print(f"{view_mat=}")
        print(f"{ndc_proj_mat=}")

        fov_x = camera_config.fov_w
        fov_y = camera_config.fov_h

        focal_x = camera_config.img_w / \
            (camera_config.foc_l + camera_config.foc_r)
        focal_y = camera_config.img_h / \
            (camera_config.foc_u + camera_config.foc_d)

        return Camera(
            camera_center=view_mat[:3, 3],

            image_height=camera_config.img_h,
            image_width=camera_config.img_w,

            FoVx=fov_x,
            FoVy=fov_y,

            focal_x=focal_x,
            focal_y=focal_y,

            world_view_transform=view_mat.transpose(0, 1),
            projection_matrix=ndc_proj_mat.transpose(0, 1),
        )

    def forward(
        self,
        *,

        camera_transform: transform_utils.ObjectTransform,
        # camera <-> world

        camera_config: camera_utils.CameraConfig,
        gp_means: torch.Tensor,  # [..., 3]
        gp_opacities: torch.Tensor,  # [..., 1]
        gp_scales: torch.Tensor,  # [..., 3]
        gp_rots: torch.Tensor,  # [..., 4]
        gp_shs: torch.Tensor,  # [..., ?]
    ):
        pc = PC(
            means=gp_means,
            opacities=gp_opacities,
            scales=gp_scales,
            rots=gp_rots,
            shs=gp_shs,
        )

        result = self.torch_splatting_gauss_renderer(
            camera=GaussianRendererWrapper.MakeCamera(
                camera_transform, camera_config),
            pc=pc,
        )

        color = result["render"].flip(0)
        alpha = result["alpha"].flip(0)
        depth = result["depth"].flip(0)
        visiility_filter = result["visiility_filter"].flip(0)
        radii = result["radii"].flip(0)

        return {
            "color": color,
            "alpha": alpha,
            "depth": depth,
            "visiility_filter": visiility_filter,
            "radii": radii,
        }


def main1():
    pytorch3d_view_dirs = utils.Dir3.FromStr("LUF")

    pytorch3d_view_coord = transform_utils.ObjectTransform.from_matching(
        pos=utils.ORIGIN,
        dirs=pytorch3d_view_dirs,
        vecs=(utils.X_AXIS, utils.Y_AXIS, utils.Z_AXIS),
    )  # camera <-> view

    radius = 10.0
    theta = 60.0 * utils.DEG
    phi = (180.0 + 45.0) * utils.DEG

    camera_transform = camera_utils.make_view(
        origin=torch.tensor(utils.sph_to_cart(radius, theta, phi),
                            dtype=utils.FLOAT),
        aim=utils.ORIGIN,
        quasi_u_dir=utils.Z_AXIS,
    )  # camera <-> world

    camera_config = camera_utils.CameraConfig.from_fov_diag(
        fov_diag=90 * utils.DEG,
        depth_near=0.1,
        depth_far=100.0,
        img_h=768,
        img_w=1280,
    )

    gauss_renderer = GaussianRendererWrapper()

    N = 3

    SH_DEG = 1

    # gp_means = torch.rand((N, 3), dtype=utils.FLOAT)
    gp_means = torch.tensor([
        # [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ], dtype=utils.FLOAT)

    gp_opacities = torch.ones((N, 1), dtype=utils.FLOAT)

    gp_scales = torch.tensor([
        # [1.0, 1.0, 1.0],
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, 0.1, 1.0],
    ], dtype=utils.FLOAT)

    # gp_rots = utils.RandQuaternion((N,), dtype=utils.FLOAT)
    gp_rots = torch.tensor([
        # [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ], dtype=utils.FLOAT)

    gp_shs = torch.rand((N, (SH_DEG + 1)**2, 3), dtype=utils.FLOAT)

    print(f"{gp_means.shape=}")
    print(f"{gp_opacities.shape=}")
    print(f"{gp_scales.shape=}")
    print(f"{gp_rots.shape=}")
    print(f"{gp_shs.shape=}")

    result = gauss_renderer(
        camera_transform=camera_transform,
        camera_config=camera_config,
        gp_means=gp_means,
        gp_opacities=gp_opacities,
        gp_scales=gp_scales,
        gp_rots=gp_rots,
        gp_shs=gp_shs,
        order="c h w",
    )

    img = result["color"]

    print(f"{type(img)=}")

    if isinstance(img, torch.Tensor):
        print(f"{img.shape=}")

    vision_utils.write_image(
        DIR / "out.png",
        img * 255,
    )


if __name__ == "__main__":
    main1()
