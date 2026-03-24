"""Progress bar utilities."""

from __future__ import annotations

from collections.abc import Iterable

from tqdm.auto import tqdm


def maybe_tqdm(
    iterable: Iterable,
    *,
    verbose: bool | int,
    level: int = 1,
    **kwargs,
):
    """Wrap an iterable with tqdm if verbose >= level, else pass through."""
    v = int(verbose) if isinstance(verbose, bool) else verbose
    if v >= level:
        return tqdm(iterable, **kwargs)
    return iterable
