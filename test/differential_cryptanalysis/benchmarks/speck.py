"""SPECK differential-search benchmark: compare MILP/SAT configurations by solving time.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this benchmarks dir, for `configs`/`framework`
import configs as C
from framework import dump_all, run_benchmark


def _cipher(rounds, version=32):
    from primitives.speck import SPECK_PERMUTATION

    return SPECK_PERMUTATION(r=rounds, version=version)


# MILP configs.
MILP_CONFIGS = [
    C.milp,                          # milp
    C.with_matsui(C.milp),           # milp_matsui
    C.modadd(C.milp, 1),             # milp_modadd1
    C.modadd(C.milp, 2),             # milp_modadd2
    C.modadd(C.milp, 3),             # milp_modadd3
    C.with_solver(C.milp, "SCIP"),   # milp_scip
]

# SAT configs.
SAT_CONFIGS = [
    C.with_solver(C.sat, "Cadical195"),                                   # sat
    C.with_solver(C.sat_increasing_atmost, "Cadical195"),                 # sat_increasing_atmost
    C.with_solver(C.sat_increasing_exact, "Cadical195"),                  # sat_increasing_exact
    C.with_solver(C.sat_adaptive, "Cadical195"),                          # sat_adaptive
    C.with_solver(C.with_matsui(C.sat_increasing_atmost), "Cadical195"),  # sat_increasing_atmost_matsui
    C.with_solver(C.with_matsui(C.sat_adaptive), "Cadical195"),           # sat_adaptive_matsui
    C.with_solver(C.atmost_encoding(C.sat, 1), "Cadical195"),             # sat_atmost1
    C.with_solver(C.atmost_encoding(C.sat, 7), "Cadical195"),             # sat_atmost7
    C.with_solver(C.atmost_encoding(C.sat, 8), "Cadical195"),             # sat_atmost8
]

def _run(backend, configs, r_start, r_end, version, time_limit, force):
    """Sweep one backend over its round range; dump ``speck<version>_<backend>.*`` into result_speck/."""
    from pathlib import Path

    out_dir = Path(__file__).with_name("result_speck")
    out_dir.mkdir(exist_ok=True)
    stem = f"speck{version}_{backend}"
    result = run_benchmark(lambda r: _cipher(r, version), configs, range(r_start, r_end + 1),
                           time_limit=time_limit,
                           cache_path=out_dir / (stem + "_progress.json"),
                           detail_path=out_dir / (stem + "_progress_detail.json"), force=force)
    dump_all(result, out_dir, stem)
    return result


# MILP and SAT usually scale to different round counts, so each has its own entry point / range.
# force=False resumes from the progress cache (skips already-solved rounds); force=True re-solves all.
def search_milp(r_start=1, r_end=2, version=32, time_limit=None, force=False):
    return _run("milp", MILP_CONFIGS, r_start, r_end, version, time_limit, force)


def search_sat(r_start=1, r_end=4, version=32, time_limit=None, force=False):
    return _run("sat", SAT_CONFIGS, r_start, r_end, version, time_limit, force)


if __name__ == "__main__":
    search_milp()
    search_sat()
