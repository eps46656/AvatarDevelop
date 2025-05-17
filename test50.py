from beartype import beartype
import collections
import typing


@beartype
class edge_padding:
    def __init__(
        self,
        it: typing.Iterable[object],
        pre_n: int,
        post_n: int,
    ):
        self.it = it
        self.pre_n = pre_n
        self.post_n = post_n

        try:
            self.length = self.pre_n + len(self.it) + self.post_n
        except:
            self.length = None

    def __len__(self) -> int:
        if self.length is None:
            raise TypeError()

        return self.length

    def __iter__(self) -> typing.Iterable[object]:
        it = iter(self.it)

        try:
            item = next(it)
        except StopIteration:
            return

        for i in range(self.pre_n):
            yield item

        yield item

        for i in it:
            item = i
            yield item

        for i in range(self.post_n):
            yield item


@beartype
class slide_window:
    def __init__(
        self,
        gen: typing.Iterable[object],
        n: int,
    ):
        self.gen = gen
        self.n = n

        try:
            self.length = max(0, len(self.gen) - self.n + 1)
        except:
            self.length = None

    @beartype
    def __len__(self) -> int:
        if self.length is None:
            raise TypeError()

        return self.length

    def __iter__(self) -> typing.Iterable[list[object]]:
        it = iter(self.gen)

        d = collections.deque()

        for i in it:
            d.append(i)

            if len(d) < self.n:
                continue

            if self.n < len(d):
                d.popleft()

            yield list(d)


@beartype
def slide_window_with_padding(
    gen: typing.Iterable[object],
    n: int,
):
    r = n // 2

    return slide_window(edge_padding(gen, r, r), n)


def main1():
    gen = slide_window_with_padding(range(6), 5)

    print(f"{len(gen)=}")

    for l in gen:
        print(l)


if __name__ == "__main__":
    main1()
