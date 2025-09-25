from collections.abc import Generator
from typing import TypeVar

T = TypeVar("T")


def batch(lst: list[T], batch_size: int) -> Generator[list[T], None, None]:
    """
    Yield successive batches of `batch_size` from `lst`.
    """
    length = len(lst)
    for i in range(0, length, batch_size):
        yield list(lst[i : i + batch_size])
