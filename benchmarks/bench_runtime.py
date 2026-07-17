"""Runtime / memory budgets for the core flow (issue #120 reference).

Wall-clock for fit / transform / score_samples and peak fit memory via
``tracemalloc``, across a few (n, d) sizes.  Numbers are laptop-class and
indicative, not a micro-benchmark; the point is the scaling shape and the
memory-cap knob.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from rbig import AnnealedRBIG

from ._report import Section, Table, fmt


def _time(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run() -> Section:
    rows = []
    for n, d in ((1000, 2), (3000, 5), (6000, 8)):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, d))
        tracemalloc.start()
        model, t_fit = _time(
            lambda: AnnealedRBIG(n_layers=12, random_state=0).fit(X)  # noqa: B023
        )
        _peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        _, t_tr = _time(lambda: model.transform(X))  # noqa: B023
        _, t_sc = _time(lambda: model.score_samples(X))  # noqa: B023
        rows.append(
            [
                f"{n}×{d}",
                str(len(model.layers_)),
                fmt(t_fit, 2),
                fmt(t_tr, 2),
                fmt(t_sc, 2),
                fmt(_peak / 1e6, 1),
            ]
        )
    table = Table(
        columns=[
            "n × d",
            "layers",
            "fit (s)",
            "transform (s)",
            "score (s)",
            "peak fit mem (MB)",
        ],
        rows=rows,
        caption=(
            "Single-thread, seed 0. Default marginals store "
            "the full sorted columns (O(layers·n·d), n_layers≤12 here); "
            "pass "
            "`MarginalGaussianize(n_quantiles=...)` via `strategy` to cap "
            "memory at O(layers·n_quantiles·d)."
        ),
    )
    return Section(
        title="Runtime and memory",
        intro="Wall-clock and peak fit memory across sizes.",
        tables=[table],
        takeaway=(
            "Fit cost is dominated by the per-layer marginal sort and the "
            "rotation; both scale roughly linearly in n and near-linearly "
            "in the layer count (early stopping usually keeps it well below "
            "the cap). Memory is the sorted-column store — use the "
            "`n_quantiles` marginal cap for large n."
        ),
    )
