"""Benchmark framework: time MILP/SAT differential search across configurations."""
import copy
import csv
import json
import time
from dataclasses import dataclass
from typing import Callable, Optional


# Keys dropped from a saved config -- not settings worth recording: runtime output/telemetry
# (``decimal_objective_function`` = live Sbox+DDT, ``resource_usage`` = CPU/RAM), and structure
# derived mechanically from (cipher, round) (``functions``/``rounds``/``layers``/``positions``/``filename``).
_CONFIG_NOISE_KEYS = (
    "decimal_objective_function", 
    "functions", "rounds", "layers", "positions", "filename",
)


def _clean_config(cfg):
    """Drop runtime-internal noise (see ``_CONFIG_NOISE_KEYS``) so the record is a clean, portable config."""
    if not isinstance(cfg, dict):
        return cfg
    return {k: v for k, v in cfg.items() if k not in _CONFIG_NOISE_KEYS}


@dataclass
class Config:
    """One benchmark configuration.

    ``build(round, prev_bounds)`` returns ``(config_model, config_solver)`` for the given round;
    ``prev_bounds`` are the earlier rounds' integer objectives (SAT) / weights (MILP), needed by
    Matsui / the SAT bound strategies. ``name`` is the snake_case identity (CSV column / result key).
    """

    name: str
    backend: str  # "milp" | "sat"
    build: Callable[[int, list], tuple]
    description: str = ""


class BenchmarkResult:
    """Weights and per-config solving times over a round range; renders CSV / rST."""

    def __init__(self, rounds, configs, data, goal=None, constraints=None, objective_target=None):
        self.rounds = list(rounds)
        self.configs = list(configs)
        self.data = data  # {config.name: {"weights", "times", "resolved", "inputs"}}
        self.goal = goal
        self.constraints = list(constraints or [])
        self.objective_target = objective_target

    def _weight(self, i):
        """The round's weight -- every config that found one must agree (all search the same optimum).

        Raises ``ValueError`` when configs disagree, since that signals a wrong / non-optimal result.
        """
        found = {cfg.name: self.data[cfg.name]["weights"][i]
                 for cfg in self.configs if self.data[cfg.name]["weights"][i] is not None}
        distinct = set(found.values())
        if len(distinct) > 1:
            raise ValueError(
                f"Configs disagree on the round {self.rounds[i]} weight (all should find the same "
                f"optimum): " + ", ".join(f"{n}={w}" for n, w in found.items())
            )
        return next(iter(distinct)) if distinct else None

    def dump_csv(self, path):
        """Wide table: Rounds, W, then one solving-time (seconds) column per config."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rounds", "W", *[c.name for c in self.configs]])
            for i, r in enumerate(self.rounds):
                writer.writerow([r, self._weight(i), *[self.data[c.name]["times"][i] for c in self.configs]])
        return path

    @staticmethod
    def _tex_escape(value):
        s = str(value)
        for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")):
            s = s.replace(a, b)
        return s

    def dump_latex(self, path=None):
        """Render the per-round time table as a LaTeX ``tabular`` (Rounds, W, one column per config).

        Escapes LaTeX-special characters (config names contain ``_``), so the output compiles as-is.
        Returns the LaTeX string; writes it to ``path`` when given.
        """
        cols = ["Rounds", "W", *[c.name for c in self.configs]]
        lines = [r"\begin{tabular}{" + "r" * len(cols) + "}", r"\hline"]
        lines.append(" & ".join(self._tex_escape(c) for c in cols) + r" \\")
        lines.append(r"\hline")
        for i, r in enumerate(self.rounds):
            row = [r, self._weight(i), *[self.data[c.name]["times"][i] for c in self.configs]]
            lines.append(" & ".join(self._tex_escape(v) for v in row) + r" \\")
        lines += [r"\hline", r"\end{tabular}"]
        text = "\n".join(lines) + "\n"
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def dump_resolved_configs_json(self, path):
        """Save the fully-resolved config_model / config_solver used for each config, per round.

        Layout: ``{config_name: [{"round", "config_model", "config_solver"}, ...]}``. Non-serializable
        leftovers are stringified as ``<TypeName>`` so the file always writes.
        """
        payload = {c.name: self.data[c.name]["resolved"] for c in self.configs}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=lambda o: f"<{type(o).__name__}>")
        return path

    def dump_config_table_rst(self, path):
        """Render the configs as an rST list-table -- the docs' config listing, now self-contained.

        Adds a ``config_model`` / ``config_solver`` column (the input the user writes) so the table
        alone documents what each config passes; the values are the last swept round's, since
        round-dependent options (Matsui ``best_obj`` / the SAT bound) advance per round.
        """
        i = len(self.rounds) - 1
        lines = [
            f".. list-table:: Benchmark configurations (``config_model`` shown at round {self.rounds[i]})",
            "   :header-rows: 1",
            "",
            "   * - Config",
            "     - Backend",
            "     - ``config_model``",
            "     - ``config_solver``",
            "     - Description",
        ]
        for c in self.configs:
            inp = self.data[c.name]["inputs"][i]
            lines += [
                f"   * - ``{c.name}``",
                f"     - {c.backend}",
                f"     - ``{inp['config_model']!r}``",
                f"     - ``{inp['config_solver']!r}``",
                f"     - {c.description}",
            ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path


def dump_all(result, out_dir, stem):
    """Write all products (``<stem>.csv``/``.tex``/``_configs.rst``/``_resolved.json``) into ``out_dir``."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    result.dump_csv(out_dir / (stem + ".csv"))
    result.dump_latex(out_dir / (stem + ".tex"))
    result.dump_config_table_rst(out_dir / (stem + "_configs.rst"))
    result.dump_resolved_configs_json(out_dir / (stem + "_resolved.json"))
    return out_dir


def run_benchmark(
    cipher_factory,
    configs,
    rounds,
    goal="DIFFERENTIALPATH_PROB",
    constraints=("INPUT_NOT_ZERO",),
    objective_target="OPTIMAL",
    time_limit: Optional[float] = None,
    cache_path=None,
    detail_path=None,
    force=False,
    verbose=True,
):
    """Run each config over ``rounds`` and return a :class:`BenchmarkResult`.

    ``cipher_factory(round)`` builds the cipher for a round count. ``time_limit`` (seconds), when
    set, is passed to the solver so one slow config cannot hang the whole sweep.

    Resume cache (split across two files): ``cache_path`` holds the at-a-glance ``{config: {round:
    {weight, time, bound}}}``; ``detail_path`` holds ``{config: {round: {resolved}}}`` (the resolved
    config, needed to rebuild resolved.json -- the only per-round output that cannot be recomputed).
    A ``(config, round)`` is reused (not re-solved) only when present in *both*. Both are written
    after each round, so an interrupted sweep loses nothing. Pass ``force=True`` to re-solve every round.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]  # framework -> benchmarks -> differential_cryptanalysis -> test -> OCP
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import attacks.attacks as attacks

    def _load(path):
        # Always load (even with force) so re-solving a sub-range preserves the other rounds; `force`
        # only gates reuse (the skip below), not what stays on disk.
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    cache = _load(cache_path)    # light: {config: {round: {weight, time, bound}}}
    detail = _load(detail_path)  # heavy: {config: {round: {resolved, input}}}

    def _dump(obj, path):
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, default=lambda o: f"<{type(o).__name__}>")

    def _persist():
        _dump(cache, cache_path)
        _dump(detail, detail_path)

    def _input(cfg, r, bounds):
        """The input config the user writes for (cfg, round) -- recomputed, never cached (it is a
        deterministic function of the config and the bound history, so storing it would be redundant)."""
        cm, cs = cfg.build(r, bounds)
        cm = {**cm, "model_type": cfg.backend}
        if time_limit is not None:
            cs = {**cs, "time_limit": time_limit}
        return {"round": r, "config_model": cm, "config_solver": cs}

    rounds = list(rounds)
    data = {}
    for cfg in configs:
        weights, times, resolved, inputs = [], [], [], []
        bounds = []  # the integer objective per round, fed to cfg.build (SAT bound / Matsui history)
        cfg_cache = cache.setdefault(cfg.name, {})
        cfg_detail = detail.setdefault(cfg.name, {})
        for r in rounds:
            lite = None if force else cfg_cache.get(str(r))
            det = None if force else cfg_detail.get(str(r))
            input_rec = _input(cfg, r, bounds)  # recomputed each round; not stored (see _input)
            inputs.append(input_rec)
            if lite is not None and det is not None:  # Resume: reuse a solved round, do not re-solve.
                weights.append(lite["weight"])
                times.append(lite["time"])
                bounds.append(lite.get("bound", lite["weight"]))
                resolved.append(det["resolved"])
                if verbose:
                    print(f"[{cfg.name}] r={r} weight={lite['weight']} (cached)")
                continue

            # diff_attacks expands config_model in place, so pass copies and keep input_rec pristine.
            config_model = copy.deepcopy(input_rec["config_model"])
            config_solver = copy.deepcopy(input_rec["config_solver"])
            start = time.perf_counter()
            trails = attacks.diff_attacks(
                cipher_factory(r),
                goal=goal,
                constraints=list(constraints),
                objective_target=objective_target,
                show_mode=0,
                config_model=config_model,
                config_solver=config_solver,
            )
            elapsed = round(time.perf_counter() - start, 3)
            weight = trails[0].data["diff_weight"] if trails else None
            # Bound history for cfg.build: the integer objective (SAT OPTIMAL sets it) so an integer
            # SAT bound starts correctly even with decimal weights; MILP / infeasible fall back to the
            # true weight (what Matsui best_obj wants).
            bound = trails[0].data.get("integer_obj_fun_value") if trails else None
            if bound is None:
                bound = weight
            if bound is None:  # infeasible / timed out: carry the last valid bound so _prev/Matsui stay usable
                bound = bounds[-1] if bounds else 0
            times.append(elapsed)
            weights.append(weight)
            bounds.append(bound)
            # The fully-resolved configs actually used (OCP fills in defaults); fall back to what we
            # passed when the search was infeasible and returned no trail.
            src = trails[0].data if trails else {"config_model": config_model, "config_solver": config_solver}
            resolved_rec = {
                "round": r,
                "config_model": _clean_config(src.get("config_model", {})),
                "config_solver": _clean_config(src.get("config_solver", {})),
            }
            resolved.append(resolved_rec)
            cfg_cache[str(r)] = {"weight": weight, "time": elapsed, "bound": bound}
            cfg_detail[str(r)] = {"resolved": resolved_rec}
            _persist()  # Persist after each round so an interrupt keeps everything solved so far.
            if verbose:
                print(f"[{cfg.name}] r={r} weight={weight} time={elapsed}s")
        data[cfg.name] = {"weights": weights, "times": times, "resolved": resolved, "inputs": inputs}
    return BenchmarkResult(rounds, configs, data, goal=goal, constraints=constraints,
                           objective_target=objective_target)
