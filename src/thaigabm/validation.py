"""Three-level validation, per guideline section 13.

    Level 1  behavioural     simulated decision rates vs Thai survey statistics
    Level 2  aggregate       simulated mobility vs an observed mobility series
    Level 3  epidemiological simulated reported cases vs an observed case series

Level 1 is the one that can always run, because the targets are baked into
evidence.py. Levels 2 and 3 need a user-supplied CSV; without one they report
"no observed data" rather than inventing a comparison.

The critical-review literature is blunt that "the behaviour looks human" is not
validation [5], so every metric here is a number, and each level records how
many simulated days actually entered the comparison.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field, asdict

from src.thaigabm.evidence import THAI_BEHAVIOURAL_TARGETS
from src.thaigabm.runner import RunResult


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def mae(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum(abs(a[i] - b[i]) for i in range(n)) / n if n else float("nan")


def rmse(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)) / n) if n else float("nan")


def mape(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    pairs = [(a[i], b[i]) for i in range(n) if b[i] > 0]
    if not pairs:
        return float("nan")
    return 100.0 * sum(abs(x - y) / y for x, y in pairs) / len(pairs)


def pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    xa, xb = a[:n], b[:n]
    ma, mb = sum(xa) / n, sum(xb) / n
    num = sum((xa[i] - ma) * (xb[i] - mb) for i in range(n))
    da = math.sqrt(sum((x - ma) ** 2 for x in xa))
    db = math.sqrt(sum((x - mb) ** 2 for x in xb))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def peak_error(sim: list[float], obs: list[float]) -> dict:
    if not sim or not obs:
        return {}
    si = max(range(len(sim)), key=lambda i: sim[i])
    oi = max(range(len(obs)), key=lambda i: obs[i])
    return {
        "peak_day_sim": si + 1, "peak_day_obs": oi + 1,
        "peak_day_error": si - oi,
        "peak_height_sim": sim[si], "peak_height_obs": obs[oi],
        "peak_height_ratio": (sim[si] / obs[oi]) if obs[oi] else float("nan"),
    }


# --------------------------------------------------------------------------
# Level 1 -- behavioural
# --------------------------------------------------------------------------
@dataclass
class LevelResult:
    level: str
    passed: bool | None
    detail: dict = field(default_factory=dict)
    note: str = ""


def level1_behavioural(result: RunResult, salience_threshold: float = 2.0,
                       tolerance: float = 0.15) -> LevelResult:
    """Compare decision rates during the high-salience period with survey rates.

    The survey was fielded during Thailand's first wave, when the outbreak was
    highly salient, so we compare against the equivalent simulated window
    (days on which reported incidence exceeded ``salience_threshold`` per
    100k) rather than against the whole run.
    """
    window = [r for r in result.history
              if r.get("reported_per_100k", 0.0) >= salience_threshold]
    if not window:
        return LevelResult("1_behavioural", None,
                           note=(f"No day reached {salience_threshold}/100k reported "
                                 "incidence; the outbreak never became salient, so "
                                 "the survey window has no simulated counterpart."))

    def m(key: str) -> float:
        return sum(r[key] for r in window) / len(window)

    sim = {"mask_use": m("mask"), "social_distancing": m("distancing")}
    detail: dict = {"n_days_in_window": len(window),
                    "salience_threshold_per_100k": salience_threshold}
    ok = True
    for key, value in sim.items():
        target = THAI_BEHAVIOURAL_TARGETS[key]
        err = value - target.value
        within = abs(err) <= tolerance
        ok &= within
        detail[key] = {"simulated": round(value, 4), "target": target.value,
                       "error": round(err, 4), "within_tolerance": within,
                       "source": target.source}
    detail["tolerance"] = tolerance
    return LevelResult("1_behavioural", ok, detail)


# --------------------------------------------------------------------------
# Level 2 / 3 -- observed-series comparison
# --------------------------------------------------------------------------
def load_observed(path: str, value_column: str, day_column: str = "day") -> list[float]:
    rows: list[tuple[float, float]] = []
    with open(path) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if value_column not in row:
                raise KeyError(f"{path} has no column {value_column!r}; "
                               f"found {list(row)}")
            try:
                d = float(row.get(day_column, i + 1))
                v = float(row[value_column])
            except (TypeError, ValueError):
                continue
            rows.append((d, v))
    rows.sort()
    return [v for _, v in rows]


def _compare(sim: list[float], obs: list[float], label: str,
             rescale: bool) -> LevelResult:
    n = min(len(sim), len(obs))
    if n == 0:
        return LevelResult(label, None, note="No overlapping days.")
    s, o = sim[:n], obs[:n]
    scale = 1.0
    if rescale and sum(s) > 0:
        # Compare *shape* when the simulated population is not the observed one.
        scale = sum(o) / sum(s)
        s = [x * scale for x in s]
    detail = {
        "n_days": n, "scale_applied": round(scale, 6),
        "MAE": mae(s, o), "RMSE": rmse(s, o), "MAPE_pct": mape(s, o),
        "pearson_r": pearson(s, o),
    }
    detail.update(peak_error(s, o))
    return LevelResult(label, None, detail,
                       note="Metrics only -- pass/fail needs a pre-registered threshold.")


def level2_mobility(result: RunResult, observed_csv: str | None,
                    column: str = "mobility", rescale: bool = False) -> LevelResult:
    if not observed_csv:
        return LevelResult("2_aggregate", None,
                           note="No observed mobility series supplied "
                                "(--observed-mobility). Skipped.")
    if not os.path.exists(observed_csv):
        return LevelResult("2_aggregate", None, note=f"File not found: {observed_csv}")
    return _compare(result.series("mobility_index"),
                    load_observed(observed_csv, column), "2_aggregate", rescale)


def level3_epidemiological(result: RunResult, observed_csv: str | None,
                           column: str = "cases", rescale: bool = True) -> LevelResult:
    if not observed_csv:
        return LevelResult("3_epidemiological", None,
                           note="No observed case series supplied "
                                "(--observed-cases). Skipped.")
    if not os.path.exists(observed_csv):
        return LevelResult("3_epidemiological", None, note=f"File not found: {observed_csv}")
    return _compare(result.series("reported_cases"),
                    load_observed(observed_csv, column), "3_epidemiological", rescale)


def validate(result: RunResult, observed_cases: str | None = None,
             observed_mobility: str | None = None) -> dict:
    levels = [
        level1_behavioural(result),
        level2_mobility(result, observed_mobility),
        level3_epidemiological(result, observed_cases),
    ]
    return {lv.level: asdict(lv) for lv in levels}


def format_validation(report: dict) -> str:
    lines = ["Validation report", "=" * 60]
    for level, r in report.items():
        status = {True: "PASS", False: "FAIL", None: "n/a"}[r["passed"]]
        lines.append(f"[{status}] {level}")
        if r["note"]:
            lines.append(f"      {r['note']}")
        for k, v in r["detail"].items():
            if isinstance(v, dict):
                lines.append(f"      {k}: sim={v['simulated']} target={v['target']} "
                             f"err={v['error']:+.3f} "
                             f"{'ok' if v['within_tolerance'] else 'OUT OF TOLERANCE'}")
            elif isinstance(v, float):
                lines.append(f"      {k}: {v:.4f}")
            else:
                lines.append(f"      {k}: {v}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Cross-arm comparison
# --------------------------------------------------------------------------
def compare_arms(results: dict[str, RunResult]) -> list[dict]:
    rows = []
    for name, res in results.items():
        hist = res.history
        if not hist:
            continue
        peak_i = max(r["infectious"] for r in hist)
        peak_day = max(range(len(hist)), key=lambda i: hist[i]["infectious"]) + 1
        rows.append({
            "arm": name,
            "attack_rate": res.meta.get("attack_rate"),
            "peak_infectious": round(peak_i),
            "peak_day": peak_day,
            "total_infections": round(hist[-1]["cumulative_infections"]),
            "mean_mobility": round(sum(r["mobility_index"] for r in hist) / len(hist), 4),
            "mean_mask": round(sum(r["mask"] for r in hist) / len(hist), 4),
            "mean_isolating": round(sum(r["isolating"] for r in hist) / len(hist), 4),
            "min_Rt": round(min(r["Rt"] for r in hist), 3),
            "days_in_lockdown": sum(1 for r in hist if r["policy"] == "lockdown"),
            "llm_calls": (res.meta.get("cache_stats") or {}).get("llm_calls", 0),
            "runtime_s": res.meta.get("runtime_seconds"),
        })
    return rows


def write_comparison(rows: list[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
