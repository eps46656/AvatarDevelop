import torch


def main2():
    a = torch.zeros((4, 4), dtype=torch.float32)

    x = torch.rand((2, 2), dtype=torch.float32, requires_grad=True)

    a[:2, :2] = x

    print(f"{a=}")

    loss = a.sum()
    loss.backward()

    print(x.grad)


if __name__ == "__main__":
    main2()
