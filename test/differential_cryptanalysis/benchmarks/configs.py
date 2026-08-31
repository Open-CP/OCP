"""Reusable benchmark configuration building blocks (base atoms + composable combinators)."""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this benchmarks dir, for `framework`
from framework import Config


def static(config_model, config_solver=None):
    """A build that always returns the same (config_model, config_solver)."""
    cm, cs = dict(config_model), dict(config_solver or {})
    return lambda r, w: (dict(cm), dict(cs))


# --------------------------- base atoms --------------------------- #
# The single MILP baseline (Gurobi, default encoding). A different solver is layered via
# ``with_solver(milp, "SCIP")``; the SAT baselines are the ``sat(solver)`` factory below.
milp = Config("milp", "milp", static({}), "Baseline MILP (Gurobi, default encoding).")


def _prev(w, k=1):
    """The k-th previous round's SAT bound as an int (0 before enough rounds have run).

    ``w`` is the bound history from ``run_benchmark`` -- the integer objective (``integer_obj_fun_value``)
    for SAT, so this is already integral.
    """
    return int(w[-k]) if len(w) >= k else 0


# SAT baseline atoms: like ``milp`` they carry no solver -- each cipher layers one via
# ``with_solver(sat, "Cadical195")``. ``sat`` is the default objective search; the others set a
# bound strategy whose start comes from the previous round's integer objective (see ``_prev``).
sat = Config("sat", "sat", static({}), "SAT, default objective search.")


def _sat_increasing_atmost(r, w):
    return {"optimal_search_strategy_sat": f"INCREASING FROM AT MOST {_prev(w)}"}, {}


def _sat_increasing_exact(r, w):
    return {"optimal_search_strategy_sat": f"INCREASING FROM EXACTLY {_prev(w)}"}, {}


def _sat_adaptive(r, w):
    # Extrapolate the next bound from the recent trend: prev + last per-round increment (= prev when <2 rounds).
    start = _prev(w) + (_prev(w, 1) - _prev(w, 2) if len(w) >= 2 else 0)
    return {"optimal_search_strategy_sat": f"ADAPTIVE FROM AT MOST {start}"}, {}


sat_increasing_atmost = Config("sat_increasing_atmost", "sat", _sat_increasing_atmost, "SAT, increasing AT-MOST bound.")
sat_increasing_exact = Config("sat_increasing_exact", "sat", _sat_increasing_exact, "SAT, increasing EXACTLY bound.")
sat_adaptive = Config("sat_adaptive", "sat", _sat_adaptive, "SAT, adaptive bound.")


# --------------------------- combinators --------------------------- #
def with_matsui(cfg):
    """Layer Matsui branch-and-bound (a backend-agnostic model constraint) onto ``cfg``.

    Matsui needs the best weights of all rounds 1..r-1, so a Matsui benchmark must start at round 1
    (``best_obj`` must have length ``Round - 1``); it is skipped for rounds whose full history is
    not present.
    """

    def build(r, w):
        cm, cs = cfg.build(r, w)
        if r >= 2 and len(w) == r - 1:
            # w is the per-round bound history: integer objective for SAT, true weight for MILP.
            cm = {**cm, "matsui_constraint": {"Round": r, "best_obj": list(w)}}
        return cm, cs

    return replace(cfg, name=f"{cfg.name}_matsui", build=build, description=(cfg.description + " +Matsui").strip())


def modadd(cfg, version):
    """Use ModAdd encoding ``XORDIFF_{version}`` (v=1/2/3) on top of ``cfg``."""

    def build(r, w):
        cm, cs = cfg.build(r, w)
        cm = {**cm, "model_version": {"ModAdd": f"XORDIFF_{version}"}}
        return cm, cs

    return replace(cfg, name=f"{cfg.name}_modadd{version}", build=build, description=(cfg.description + f" +ModAdd v{version}").strip())


def atmost_encoding(cfg, n):
    """Use SAT at-most cardinality encoding ``n`` (e.g. 1/7/8) on top of ``cfg``."""

    def build(r, w):
        cm, cs = cfg.build(r, w)
        cm = {**cm, "atmost_encoding_sat": n}
        return cm, cs

    return replace(cfg, name=f"{cfg.name}_atmost{n}", build=build, description=(cfg.description + f" +at-most-enc {n}").strip())


def sbox_tool(cfg, sbox_name, tool_type, label=None):
    """Model S-box ``sbox_name`` with a specific tool (e.g. ``"polyhedron"``) on top of ``cfg``.

    The S-box modeling axis for S-box ciphers, analogous to ``modadd`` for ARX. ``label`` overrides
    the config-name suffix (defaults to ``tool_type``; use a short tag for long tool names).
    """

    def build(r, w):
        cm, cs = cfg.build(r, w)
        cm = {**cm, "model_params": {sbox_name: {"tool_type": tool_type}}}
        return cm, cs

    return replace(cfg, name=f"{cfg.name}_{label or tool_type}", build=build, description=(cfg.description + f" +{sbox_name}:{tool_type}").strip())


def with_solver(cfg, solver):
    """Run ``cfg`` with a specific solver (e.g. ``\"SCIP\"``, ``\"Glucose3\"``)."""

    def build(r, w):
        cm, cs = cfg.build(r, w)
        cs = {**cs, "solver": solver}
        return cm, cs

    return replace(cfg, name=f"{cfg.name}_{solver.lower()}", build=build, description=(cfg.description + f" +{solver}").strip())
