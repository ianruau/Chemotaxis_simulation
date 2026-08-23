#!/usr/bin/env python3
r"""Is a global classical solution guaranteed for these parameters, in 1-D?

Chemotaxis Paper I (Chen--Ruau--Shen) gives sufficient conditions under which
the parabolic-elliptic system has a positive classical solution that is bounded
and, in the better cases, global.  Paper III fixes the space dimension to one
and studies bifurcation numerically.  A time-stepping run only means something
on a parameter tuple for which such a solution is known to exist, so it is
useful to say, at the moment a run starts, which condition covers it.

WARNING, NOT REFUSAL
--------------------
The conditions are SUFFICIENT, not necessary.  A tuple outside them is not
known to blow up; it is merely not covered by the cited results.  Running
outside the region is legitimate and in fact deliberate here -- the
unboundedness and spike-formation experiments do exactly that.  So the default
behaviour is to report, never to stop.  Pass ``require=True`` (or
``--require-admissible`` on the command line) only when a hard gate is wanted,
for instance in a reproducibility run.

The conditions, specialized to N = 1
------------------------------------
Write (BDD) for ``limsup_t ||u(t)||_inf < infinity``.

Non-minimal model, ``a, b > 0`` and ``m > 0``.  For ``alpha <= 2``, (BDD) holds
when ``alpha >= m + gamma - 1`` for any ``beta >= 0``, and also when
``beta >= 1/2`` and ``alpha >= 2m + gamma - 2``.  If additionally ``m >= 1``
then the solution is global, and since ``m + gamma - 1 <= 2m + gamma - 2``
there, the region is just ``alpha >= m + gamma - 1``.

The two equality cases carry a smallness threshold on ``chi_0`` whose
denominator contains ``(N alpha - 2)_+``, read as ``+infinity`` when that factor
vanishes.  At ``N = 1`` the factor is ``(alpha - 2)_+``, so for ``alpha <= 2``
the threshold is infinite and ``chi_0`` is unconstrained -- which is what admits
the large ``chi_0`` used near the bifurcation threshold.  For ``alpha > 2`` the
threshold is finite and depends on constants (``K``, ``Psi_beta``,
``Theta_beta``) that this module does not reproduce, so that case is reported
as UNKNOWN rather than guessed.

Minimal model, ``a = b = 0``.  The logistic terms vanish, so ``alpha`` does not
enter.  (BDD) holds when ``chi_0 <= 0``, and also when ``beta >= 1`` and
``0 < m <= 1``, subject for ``m = 1`` to
``chi_0 < 2(2 beta - 1)/max{2, gamma}``.  For ``chi_0 > 0`` the solution is
global only through that last branch.  Note that boundedness alone does not
give global existence: the local theory permits ``T_max < infinity`` with
``inf u -> 0``.

This mirrors ``codes/global_existence_legality.py`` in the manuscript
repository, which applies the same rules to the tuples appearing in the paper.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

GLOBAL = "GLOBAL"
BOUNDED = "BOUNDED"
NOT_COVERED = "NOT COVERED"
UNKNOWN = "UNKNOWN"


class Verdict(NamedTuple):
    status: str
    basis: str

    @property
    def is_global(self) -> bool:
        return self.status == GLOBAL

    def line(self) -> str:
        return f"admissibility: {self.status} -- {self.basis}"


def admissibility(
    *,
    a: float,
    b: float,
    alpha: float,
    m: float,
    beta: float,
    gamma: float,
    chi: Optional[float] = None,
) -> Verdict:
    """Classify one parameter tuple against the N = 1 conditions of Paper I."""
    if m <= 0:
        return Verdict(UNKNOWN, "m must be positive")

    if a == 0 and b == 0:
        if chi is not None and chi <= 0:
            if m >= 1:
                return Verdict(GLOBAL, "minimal model, chi_0 <= 0, m >= 1")
            return Verdict(BOUNDED, "minimal model, chi_0 <= 0, but m < 1")
        if beta < 1:
            return Verdict(
                NOT_COVERED, f"minimal model needs beta >= 1 for chi_0 > 0, got beta = {beta:g}"
            )
        if m < 1:
            return Verdict(BOUNDED, "minimal model, 0 < m < 1; global existence not implied")
        if m > 1:
            return Verdict(
                NOT_COVERED, f"minimal model with m = {m:g} > 1 and chi_0 > 0 is not covered"
            )
        bound = 2.0 * (2.0 * beta - 1.0) / max(2.0, gamma)
        if chi is None:
            return Verdict(UNKNOWN, f"minimal model, m = 1: needs chi_0 < {bound:g}, chi_0 not given")
        if chi < bound:
            return Verdict(GLOBAL, f"minimal model, m = 1, chi_0 = {chi:g} < {bound:g}")
        return Verdict(
            NOT_COVERED, f"minimal model, m = 1: needs chi_0 < {bound:g}, run uses {chi:g}"
        )

    if a > 0 and b > 0:
        if alpha > 2:
            return Verdict(
                UNKNOWN,
                f"alpha = {alpha:g} > 2: the equality cases carry a finite chi_0 threshold "
                "involving constants this module does not reproduce",
            )
        wide = alpha >= m + gamma - 1
        narrow = beta >= 0.5 and alpha >= 2.0 * m + gamma - 2.0
        if not (wide or narrow):
            detail = f"alpha = {alpha:g} < m + gamma - 1 = {m + gamma - 1:g}"
            if beta >= 0.5:
                detail += f" and < 2m + gamma - 2 = {2 * m + gamma - 2:g}"
            else:
                detail += " (beta < 1/2, so the second alternative is unavailable)"
            return Verdict(NOT_COVERED, detail)
        which = (
            f"alpha >= m + gamma - 1 = {m + gamma - 1:g}"
            if wide
            else f"beta >= 1/2 and alpha >= 2m + gamma - 2 = {2 * m + gamma - 2:g}"
        )
        if m >= 1:
            return Verdict(GLOBAL, f"{which}, m >= 1")
        return Verdict(BOUNDED, f"{which}, but m < 1 so global existence is not implied")

    return Verdict(
        UNKNOWN, f"a = {a:g}, b = {b:g}: only a = b = 0 and a, b > 0 are covered"
    )


def report(verdict: Verdict, *, require: bool = False) -> None:
    """Print the verdict, and raise SystemExit only if a hard gate was requested."""
    print(verdict.line())
    if verdict.status in (NOT_COVERED, UNKNOWN):
        print(
            "  note: these are sufficient conditions, so this is not a prediction "
            "of blow-up; the run is simply not covered by Paper I."
        )
    if require and not verdict.is_global:
        raise SystemExit(
            "refusing to run: --require-admissible was given and global existence "
            f"is not guaranteed ({verdict.status})"
        )
