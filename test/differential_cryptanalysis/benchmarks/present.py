"""PRESENT differential-search benchmark: compare MILP/SAT configurations by solving time."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this benchmarks dir, for `configs`/`framework`
import configs as C
from framework import dump_all, run_benchmark

_GOAL = "DIFFERENTIALPATH_PROB"  # as in the archive search_diff_present2
_SBOX = "PRESENT_Sbox"


def _cipher(rounds):
    from primitives.present import PRESENT_PERMUTATION

    return PRESENT_PERMUTATION(r=rounds)


# PRESENT's S-box modeling axis: espresso (minimize_logic) vs polyhedron.
def _espresso(cfg):
    return C.sbox_tool(cfg, _SBOX, "minimize_logic_espresso", label="espresso")


def _poly(cfg):
    return C.sbox_tool(cfg, _SBOX, "polyhedron")


# MILP: 6 configs (archive present2 milp_1..milp_6) -- default / espresso / polyhedron, each also +Matsui.
MILP_CONFIGS = [
    C.milp,                             # milp
    _espresso(C.milp),                  # milp_espresso
    _poly(C.milp),                      # milp_polyhedron
    C.with_matsui(C.milp),              # milp_matsui
    _espresso(C.with_matsui(C.milp)),   # milp_matsui_espresso
    _poly(C.with_matsui(C.milp)),       # milp_matsui_polyhedron
]

# SAT: 9 configs (archive present2 sat_1..sat_9); the solver is layered on each via with_solver.
SAT_SOLVER = "Cadical195"
SAT_CONFIGS = [
    C.with_solver(C.sat, SAT_SOLVER),                                              # sat
    C.with_solver(_espresso(C.sat), SAT_SOLVER),                                   # sat_espresso
    C.with_solver(C.sat_increasing_atmost, SAT_SOLVER),                            # sat_increasing_atmost
    C.with_solver(C.sat_increasing_exact, SAT_SOLVER),                             # sat_increasing_exact
    C.with_solver(C.sat_adaptive, SAT_SOLVER),                                     # sat_adaptive
    C.with_solver(C.with_matsui(C.sat_increasing_atmost), SAT_SOLVER),             # sat_increasing_atmost_matsui
    C.with_solver(_espresso(C.with_matsui(C.sat_increasing_atmost)), SAT_SOLVER),  # sat_increasing_atmost_matsui_espresso
    C.with_solver(C.with_matsui(C.sat_adaptive), SAT_SOLVER),                      # sat_adaptive_matsui
    C.with_solver(_espresso(C.with_matsui(C.sat_adaptive)), SAT_SOLVER),           # sat_adaptive_matsui_espresso
]


def _run(backend, configs, r_start, r_end, time_limit, force):
    """Sweep one backend over its round range; dump ``present_<backend>.*`` into result_present/."""
    from pathlib import Path

    out_dir = Path(__file__).with_name("result_present")
    out_dir.mkdir(exist_ok=True)
    stem = f"present_{backend}"
    result = run_benchmark(lambda r: _cipher(r), configs, range(r_start, r_end + 1), goal=_GOAL,
                           time_limit=time_limit,
                           cache_path=out_dir / (stem + "_progress.json"),
                           detail_path=out_dir / (stem + "_progress_detail.json"), force=force)
    dump_all(result, out_dir, stem)
    return result


# force=False resumes from the progress cache (skips already-solved rounds); force=True re-solves all.
def search_milp(r_start=1, r_end=2, time_limit=None, force=False):
    return _run("milp", MILP_CONFIGS, r_start, r_end, time_limit, force)


def search_sat(r_start=1, r_end=2, time_limit=None, force=False):
    return _run("sat", SAT_CONFIGS, r_start, r_end, time_limit, force)


if __name__ == "__main__":
    search_milp()
    search_sat()
