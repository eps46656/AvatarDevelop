

import sympy
import torch
import utils

import camera_utils


def main1():
    ya, yb, yc, yd, x = sympy.symbols("ya yb yc yd x")

    xa = -1
    xb = 0
    xc = 1
    xd = 2

    p = \
        ya*(x - xb)*(x - xc)*(x - xd)/((xa - xb)*(xa - xc)*(xa - xd)) + \
        yb*(x - xa)*(x - xc)*(x - xd)/((xb - xa)*(xb - xc)*(xb - xd)) + \
        yc*(x - xa)*(x - xb)*(x - xd)/((xc - xa)*(xc - xb)*(xc - xd)) + \
        yd*(x - xa)*(x - xb)*(x - xc)/((xd - xa)*(xd - xb)*(xd - xc))

    p = sympy.expand(p) * 6

    print(sympy.pretty(p))


def main2():
    view_axis = "rdf"
    ndc_axis = "rdf"

    proj_mat = camera_utils.MakePersProjMat(
        view_axes="rdf",
        image_shape=(1280, 720),
        ndc_axes="rdf",
        diag_fov=45*utils.DEG,
        far=100.0,
        dtype=torch.float,
        device=utils.CPU_DEVICE
    )

    print(proj_mat)


if __name__ == "__main__":
    main2()
