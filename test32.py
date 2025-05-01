import sympy
import sympy.vector


def declare_vector(R, x, y, z):
    sym_x, sym_y, sym_z = sympy.symbols(f"{x} {y} {z}")
    return sym_x * R.x + sym_y * R.y + sym_z * R.z


def main1():
    R = sympy.vector.CoordSys3D('R')

    va = declare_vector(R, "vax", "vay", "vaz")
    vb = declare_vector(R, "vbx", "vby", "vbz")
    vc = declare_vector(R, "vcx", "vcy", "vcz")

    print(va)
    print(vb)
    print(vc)

    g = (va + vb + vc) / 3

    print(g)

    vab = sympy.simplify(vb - va)
    vac = sympy.simplify(vc - va)

    z = sympy.vector.cross(vab, vac)

    print(f"{vab=}")
    print(f"{vac=}")

    print(f"{z=}")

    x = va - g

    y = sympy.vector.cross(z, x)

    print(f"{x=}")
    print(f"{y=}")
    print(f"{z=}")


if __name__ == "__main__":
    main1()
