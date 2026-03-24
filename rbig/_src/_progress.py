"""Progress bar utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def maybe_tqdm(
    iterable: Iterable[T],
    *,
    verbose: bool | int,
    level: int = 1,
    **kwargs,
) -> Iterable[T]:
    """Wrap an iterable with tqdm if verbose >= level, else pass through."""
    v = int(verbose) if isinstance(verbose, bool) else verbose
    if v >= level:
        return tqdm(iterable, **kwargs)
    return iter(iterable)
