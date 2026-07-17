"""Tiny markdown-report accumulator shared by the benchmark modules.

Each benchmark returns a :class:`Section` (title, prose, one or more
tables); ``run_all`` concatenates them into ``docs/benchmarks.md``.  The
honest-reporting gate (issue #133) is enforced by convention: every table
shows the baselines alongside RBIG, including the rows where a baseline
wins, and the prose names those rows explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Table:
    """A single markdown table: header row + string cells."""

    columns: list[str]
    rows: list[list[str]]
    caption: str = ""

    def render(self) -> str:
        head = "| " + " | ".join(self.columns) + " |"
        sep = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = "\n".join("| " + " | ".join(r) + " |" for r in self.rows)
        out = "\n".join([head, sep, body])
        if self.caption:
            out += f"\n\n*{self.caption}*"
        return out


@dataclass
class Section:
    """A titled report section: prose intro, tables, and a takeaway."""

    title: str
    intro: str = ""
    tables: list[Table] = field(default_factory=list)
    takeaway: str = ""

    def render(self, level: int = 2) -> str:
        parts = [f"{'#' * level} {self.title}"]
        if self.intro:
            parts.append(self.intro.strip())
        for table in self.tables:
            parts.append(table.render())
        if self.takeaway:
            parts.append(f"**Takeaway.** {self.takeaway.strip()}")
        return "\n\n".join(parts)


def fmt(x: float, nd: int = 3) -> str:
    """Fixed-precision float cell."""
    return f"{x:.{nd}f}"


def winner(values: dict[str, float], higher_is_better: bool = True) -> str:
    """Return the key of the best value (for bolding in prose)."""
    return (max if higher_is_better else min)(values, key=values.get)
