"""SPECK related-key differential-search benchmark on the block cipher (key schedule carries the diff)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this benchmarks dir
import configs as C
from framework import dump_all, run_benchmark


def _cipher(rounds, version=(32, 64)):
    from primitives.speck import SPECK_BLOCKCIPHER

    return SPECK_BLOCKCIPHER(r=rounds, version=list(version))  # version = [block_size, key_size]


# Own config lists so the related-key sweep can diverge from single-key (names auto-compose).
MILP_CONFIGS = [
    C.milp,                          # milp
    C.with_matsui(C.milp),           # milp_matsui
    C.modadd(C.milp, 1),             # milp_modadd1
    C.modadd(C.milp, 2),             # milp_modadd2
    C.modadd(C.milp, 3),             # milp_modadd3
    C.with_solver(C.milp, "SCIP"),   # milp_scip
]

SAT_SOLVER = "Cadical195"
SAT_CONFIGS = [
    C.with_solver(C.sat, "Cadical195"),                                   # sat
    C.with_solver(C.sat_increasing_atmost, "Cadical195"),                 # sat_increasing_atmost
    C.with_solver(C.sat_increasing_exact, "Cadical195"),                  # sat_increasing_exact
    C.with_solver(C.sat_adaptive, "Cadical195"),                          # sat_adaptive
    C.with_solver(C.with_matsui(C.sat_increasing_atmost), "Cadical195"),  # sat_increasing_atmost_matsui
    C.with_solver(C.with_matsui(C.sat_adaptive), "Cadical195"),           # sat_adaptive_matsui
]


def _run(backend, configs, r_start, r_end, version, time_limit, force):
    from pathlib import Path

    out_dir = Path(__file__).with_name("result_speck")
    out_dir.mkdir(exist_ok=True)
    tag = "_".join(str(v) for v in version)  # e.g. (32, 64) -> "32_64"
    stem = f"speck{tag}_rk_{backend}"
    result = run_benchmark(lambda r: _cipher(r, version), configs, range(r_start, r_end + 1),
                           time_limit=time_limit,
                           cache_path=out_dir / (stem + "_progress.json"),
                           detail_path=out_dir / (stem + "_progress_detail.json"), force=force)
    dump_all(result, out_dir, stem)
    return result


# force=False resumes from the progress cache (skips already-solved rounds); force=True re-solves all.
def search_rk_milp(r_start=1, r_end=2, version=(32, 64), time_limit=None, force=False):
    return _run("milp", MILP_CONFIGS, r_start, r_end, version, time_limit, force)


def search_rk_sat(r_start=1, r_end=2, version=(32, 64), time_limit=None, force=False):
    return _run("sat", SAT_CONFIGS, r_start, r_end, version, time_limit, force)


if __name__ == "__main__":
    search_rk_milp()
    search_rk_sat()
