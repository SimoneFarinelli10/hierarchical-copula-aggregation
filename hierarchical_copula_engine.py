from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import csv
from datetime import date, datetime

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau, norm

FloatArray = NDArray[np.float64]
CopulaSampler = Callable[[int, int, np.random.Generator], FloatArray]


# ---------------------------------------------------------------------------
# Copula calibration utilities
# ---------------------------------------------------------------------------

def gaussian_rho_from_kendall_tau(tau: float) -> float:
    """Map Kendall's tau to the Gaussian copula correlation parameter."""
    if not (-1.0 < tau < 1.0):
        raise ValueError("tau must lie in (-1, 1)")
    return float(np.sin(0.5 * np.pi * tau))


def mirrored_clayton_theta_from_kendall_tau(tau: float) -> float:
    """
    Map Kendall's tau to the mirrored (survival) Clayton parameter.

    For the standard Clayton copula one has tau = theta / (theta + 2), theta > 0.
    The same relation holds for the survival Clayton family.
    """
    if not (0.0 < tau < 1.0):
        raise ValueError("tau must lie in (0, 1) for mirrored Clayton")
    return float(2.0 * tau / (1.0 - tau))


def empirical_kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Estimate Kendall's tau from two equally long series."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        raise ValueError("Need at least two finite paired observations")
    tau, _ = kendalltau(x_arr[mask], y_arr[mask], nan_policy="omit")
    if tau is None or np.isnan(tau):
        raise ValueError("Unable to estimate Kendall's tau")
    return float(tau)


def aligned_log_returns_and_differences(
        price_series: Sequence[float],
        spread_series: Sequence[float],
) -> Tuple[FloatArray, FloatArray]:
    """
    Convert a market price index and a credit spread index into aligned loss-style inputs.

    Returns:
        market_losses = - log(S_t / S_{t-1})
        credit_losses = Delta OAS_t = OAS_t - OAS_{t-1}
    """
    prices = np.asarray(price_series, dtype=float)
    spreads = np.asarray(spread_series, dtype=float)

    if prices.ndim != 1 or spreads.ndim != 1:
        raise ValueError("Both inputs must be one-dimensional")
    if len(prices) != len(spreads):
        raise ValueError("Both inputs must have the same length")
    if len(prices) < 2:
        raise ValueError("Need at least two observations")

    market_losses = -np.diff(np.log(prices))
    credit_losses = np.diff(spreads)
    return market_losses.astype(float), credit_losses.astype(float)


def read_fred_csv(path: str | Path, value_column: Optional[str] = None) -> FloatArray:
    """
    Read a FRED CSV exported manually by the user and return only the values.

    This convenience wrapper drops the DATE column; use
    ``read_fred_csv_with_dates`` when date-aware alignment or windowing is needed.
    """
    _, values = read_fred_csv_with_dates(path, value_column)
    return values


def read_fred_csv_with_dates(path: str, value_column: str | None = None):
    dates = []
    values = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        if not fieldnames:
            raise ValueError(f"No header found in {path}")

        # Detect date column
        if "observation_date" in fieldnames:
            date_column = "observation_date"
        elif "DATE" in fieldnames:
            date_column = "DATE"
        else:
            raise ValueError(
                f"Could not infer date column in {path}. "
                f"Available columns: {fieldnames}"
            )

        # Detect value column
        if value_column is not None:
            if value_column not in fieldnames:
                raise ValueError(
                    f"Requested value column '{value_column}' not found in {path}. "
                    f"Available columns: {fieldnames}"
                )
            chosen_value_column = value_column
        else:
            other_cols = [c for c in fieldnames if c != date_column]
            if len(other_cols) != 1:
                raise ValueError(
                    f"Could not infer value column in {path}. "
                    f"Available columns: {fieldnames}"
                )
            chosen_value_column = other_cols[0]

        for row in reader:
            d = row[date_column].strip()
            v = row[chosen_value_column].strip()

            if d == "" or v == "" or v in {".", "NA", "NaN"}:
                continue

            dates.append(datetime.strptime(d, "%Y-%m-%d").date())
            values.append(float(v))

    print(f"Loaded {len(values)} observations from {path}")

    if len(values) < 2:
        raise ValueError(f"Not enough observations in {path}")

    return np.array(dates), np.array(values, dtype=float)


def align_series_on_common_dates(
        market_dates: Sequence[date],
        market_values: Sequence[float],
        credit_dates: Sequence[date],
        credit_values: Sequence[float],
) -> Tuple[List[date], FloatArray, FloatArray]:
    """Align two dated series on their common dates."""
    market_map = {d: float(v) for d, v in zip(market_dates, market_values)}
    credit_map = {d: float(v) for d, v in zip(credit_dates, credit_values)}
    common_dates = sorted(set(market_map) & set(credit_map))
    if len(common_dates) < 2:
        raise ValueError("Need at least two common dated observations")
    market = np.asarray([market_map[d] for d in common_dates], dtype=float)
    credit = np.asarray([credit_map[d] for d in common_dates], dtype=float)
    return common_dates, market, credit


def last_n_years_window(dates: Sequence[date], years: int = 5) -> Tuple[date, date]:
    """Return an inclusive trailing calendar window based on the sample end date."""
    if not dates:
        raise ValueError("dates must not be empty")
    end = max(dates)
    start = date(end.year - years, end.month, end.day)
    return start, end


def slice_window(common_dates, market_prices, credit_spreads, start=None, end=None):
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d").date()
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d").date()

    # Make all three indexable by numpy arrays
    common_dates = np.array(common_dates, dtype=object)
    market_prices = np.array(market_prices, dtype=float)
    credit_spreads = np.array(credit_spreads, dtype=float)

    keep_idx = []
    for i, d in enumerate(common_dates):
        if start is not None and d < start:
            continue
        if end is not None and d > end:
            continue
        keep_idx.append(i)

    if len(keep_idx) < 2:
        raise ValueError("Date window leaves fewer than two paired observations")

    keep_idx = np.array(keep_idx, dtype=int)

    return common_dates[keep_idx], market_prices[keep_idx], credit_spreads[keep_idx]


def calibrate_bivariate_pair_from_fred_csv(
        market_csv: str | Path,
        credit_csv: str | Path,
        market_column: Optional[str] = None,
        credit_column: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        window_label: str = "full sample",
) -> Dict[str, float | str]:
    """
    Calibrate from two manually downloaded FRED CSV files aligned on common dates.

    The calibrated dependence is computed from loss-style daily observations:
        market_losses = -log(S_t / S_{t-1})
        credit_losses = Delta OAS_t
    """
    market_dates, market_prices = read_fred_csv_with_dates(market_csv, market_column)
    credit_dates, credit_spreads = read_fred_csv_with_dates(credit_csv, credit_column)

    common_dates, market_prices, credit_spreads = align_series_on_common_dates(
        market_dates, market_prices, credit_dates, credit_spreads
    )

    common_dates, market_prices, credit_spreads = slice_window(
        common_dates, market_prices, credit_spreads, start=start, end=end
    )

    market_losses, credit_losses = aligned_log_returns_and_differences(
        market_prices, credit_spreads
    )
    tau = empirical_kendall_tau(market_losses, credit_losses)

    result: Dict[str, float | str] = {
        "window": window_label,
        "start_date": common_dates[0].isoformat(),
        "end_date": common_dates[-1].isoformat(),
        "n_obs": float(len(market_losses)),
        "kendall_tau": tau,
        "gaussian_rho": gaussian_rho_from_kendall_tau(tau),
    }
    if tau > 0:
        result["mirrored_clayton_theta"] = mirrored_clayton_theta_from_kendall_tau(tau)
    else:
        result["mirrored_clayton_theta"] = float("nan")
    return result


def calibrate_standard_windows_from_fred_csv(
        market_csv: str,
        credit_csv: str,
        market_column: str | None = None,
        credit_column: str | None = None,
):
    market_dates, market_prices = read_fred_csv_with_dates(market_csv, market_column)
    credit_dates, credit_spreads = read_fred_csv_with_dates(credit_csv, credit_column)

    latest_common = min(market_dates[-1], credit_dates[-1])
    earliest_common = max(market_dates[0], credit_dates[0])

    window_specs = [
        ("Full sample (since 2016)", earliest_common.isoformat(), latest_common.isoformat()),
        ("COVID crisis 2020", "2020-02-19", "2020-04-30"),
    ]

    results = []

    for label, start, end in window_specs:
        try:
            result = calibrate_bivariate_pair_from_fred_csv(
                market_csv=market_csv,
                credit_csv=credit_csv,
                market_column=market_column,
                credit_column=credit_column,
                start=start,
                end=end,
                window_label=label,
            )
            results.append(result)
        except ValueError as e:
            results.append({
                "window_label": label,
                "start": start,
                "end": end,
                "n_pairs": 0,
                "kendall_tau": None,
                "gaussian_rho": None,
                "mirrored_clayton_theta": None,
                "status": f"unavailable: {e}",
            })

    return results


def format_calibration_results_as_latex_rows(results):
    rows = []
    for r in results:
        rows.append(
            f"{r['window']} & {r['start_date']} & {r['end_date']} & "
            f"{int(r['n_obs'])} & {r['kendall_tau']:.4f} & "
            f"{r['gaussian_rho']:.4f} & {r['mirrored_clayton_theta']:.4f} \\\\"
        )
    return "\n".join(rows)


def gaussian_copula_sampler(
        n: int,
        dim: int,
        rng: np.random.Generator,
        corr: FloatArray,
) -> FloatArray:
    """Sample U ~ Gaussian copula with correlation matrix ``corr``."""
    if corr.shape != (dim, dim):
        raise ValueError(f"corr must be shape ({dim}, {dim})")

    z = rng.multivariate_normal(mean=np.zeros(dim), cov=corr, size=n)
    u = norm.cdf(z)

    eps = np.finfo(float).eps
    return np.clip(u, eps, 1.0 - eps)


def mirrored_clayton_copula_sampler(
        n: int,
        dim: int,
        rng: np.random.Generator,
        theta: float,
) -> FloatArray:
    """Sample U ~ mirrored (survival) Clayton copula with parameter theta > 0."""
    if theta <= 0:
        raise ValueError("theta must be > 0")

    v = rng.gamma(shape=1.0 / theta, scale=1.0, size=n)
    e = rng.exponential(scale=1.0, size=(n, dim))

    u = (1.0 + (e / v[:, None])) ** (-1.0 / theta)
    u = 1.0 - u

    eps = np.finfo(float).eps
    return np.clip(u, eps, 1.0 - eps)


def exchangeable_gaussian_sampler(rho: float) -> CopulaSampler:
    """Create a Gaussian copula sampler with exchangeable correlation rho."""

    def _sampler(n: int, dim: int, rng: np.random.Generator) -> FloatArray:
        if not (-1.0 / max(dim - 1, 1) < rho < 1.0):
            raise ValueError("rho is outside the admissible exchangeable range")
        corr = np.full((dim, dim), rho, dtype=float)
        np.fill_diagonal(corr, 1.0)
        return gaussian_copula_sampler(n, dim, rng, corr)

    return _sampler


def exchangeable_mirrored_clayton_sampler(theta: float) -> CopulaSampler:
    """Create a mirrored Clayton copula sampler with scalar parameter theta."""

    def _sampler(n: int, dim: int, rng: np.random.Generator) -> FloatArray:
        return mirrored_clayton_copula_sampler(n, dim, rng, theta)

    return _sampler


def simulate_from_calibrated_copula(
        rho: float,
        theta: float,
        copula_type: str,
        n: int = 10000,
        seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    # Use symmetric 2-level structure
    tau = 0.3  # placeholder internal dependence (structure assumption)

    if copula_type == "gaussian":
        sampler_root = exchangeable_gaussian_sampler(rho)
        sampler_child = exchangeable_gaussian_sampler(rho)

        tree = make_two_level_example_tree(
            root_sampler=sampler_root,
            p1_sampler=sampler_child,
            p2_sampler=sampler_child,
            root_family="Gaussian",
            p1_family="Gaussian",
            p2_family="Gaussian",
            root_param=rho,
            p1_param=rho,
            p2_param=rho,
        )

    elif copula_type == "clayton":
        sampler_root = exchangeable_mirrored_clayton_sampler(theta)
        sampler_child = exchangeable_mirrored_clayton_sampler(theta)

        tree = make_two_level_example_tree(
            root_sampler=sampler_root,
            p1_sampler=sampler_child,
            p2_sampler=sampler_child,
            root_family="Mirrored Clayton",
            p1_family="Mirrored Clayton",
            p2_family="Mirrored Clayton",
            root_param=theta,
            p1_param=theta,
            p2_param=theta,
        )

    else:
        raise ValueError("Unknown copula type")

    samples = make_example_samples(rng, n)
    aligned = apply_node_aggregation_recursively(tree, samples, rng)
    losses = aggregate_by_prefix(aligned, "root/")

    return risk_statistics(losses)


# ---------------------------------------------------------------------------
# Tree engine
# ---------------------------------------------------------------------------

@dataclass
class Node:
    name: str
    children: List["Node"] = field(default_factory=list)
    copula: Optional[CopulaSampler] = None
    copula_family: Optional[str] = None
    kendall_tau: Optional[float] = None
    parameter: Optional[float] = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0


def reorder_subtree_by_node_aggregation(
        leaf_samples: Dict[str, FloatArray],
        parent_u: FloatArray,
) -> Dict[str, FloatArray]:
    """
    Reorder one internal node's child subtrees using copula uniforms.

    Expected key format inside ``leaf_samples`` at this local level:
    ``child_name/...`` or simply ``child_name`` when the children are leaves.
    """
    if not leaf_samples:
        raise ValueError("leaf_samples must not be empty")

    n = next(iter(leaf_samples.values())).shape[0]

    if parent_u.shape[0] != n:
        raise ValueError("parent_u must have same number of rows as samples")
    if any(v.shape[0] != n for v in leaf_samples.values()):
        raise ValueError("all leaf samples must have same length n")

    children: Dict[str, List[str]] = {}
    for key in leaf_samples:
        if "/" in key:
            child, _ = key.split("/", 1)
        else:
            child = key
        children.setdefault(child, []).append(key)

    child_names = sorted(children.keys())
    dim = len(child_names)

    if parent_u.shape[1] != dim:
        raise ValueError(f"parent_u has dim={parent_u.shape[1]} but found {dim} children")

    pis: Dict[str, np.ndarray] = {}
    for child in child_names:
        agg = np.zeros(n, dtype=float)
        for leaf_key in children[child]:
            agg += leaf_samples[leaf_key]
        pis[child] = np.argsort(agg, kind="mergesort")

    out: Dict[str, FloatArray] = {}
    for idx, child in enumerate(child_names):
        u = parent_u[:, idx]
        j = np.ceil(n * u).astype(int) - 1
        j = np.clip(j, 0, n - 1)
        mapped = pis[child][j]
        for leaf_key in children[child]:
            out[leaf_key] = leaf_samples[leaf_key][mapped]

    return out


def apply_node_aggregation_recursively(
        node: Node,
        samples: Dict[str, FloatArray],
        rng: np.random.Generator,
        path: str = "",
) -> Dict[str, FloatArray]:
    """Bottom-up recursive node aggregation on an arbitrary rooted tree."""
    if node.is_leaf():
        return samples

    current_path = f"{path}/{node.name}" if path else node.name

    for child in node.children:
        samples = apply_node_aggregation_recursively(child, samples, rng, current_path)

    if node.copula is None:
        raise ValueError(f'node "{node.name}" is internal but has no copula sampler')
    if not samples:
        raise ValueError("samples must not be empty")

    n = next(iter(samples.values())).shape[0]
    dim = len(node.children)
    u = node.copula(n, dim, rng)

    prefix = current_path + "/"
    view: Dict[str, FloatArray] = {}
    rest: Dict[str, FloatArray] = {}

    for key, value in samples.items():
        if key.startswith(prefix):
            view[key[len(prefix):]] = value
        else:
            rest[key] = value

    if not view:
        raise ValueError(f'No samples found under prefix "{prefix}"')

    reordered = reorder_subtree_by_node_aggregation(view, u)

    out = dict(rest)
    for key, value in reordered.items():
        out[prefix + key] = value

    return out


def aggregate_by_prefix(samples: Dict[str, FloatArray], prefix: str) -> FloatArray:
    """Aggregate all leaf series below ``prefix``."""
    keys = [k for k in samples if k.startswith(prefix)]
    if not keys:
        raise ValueError(f"No leaves found below prefix '{prefix}'")
    total = np.zeros_like(samples[keys[0]])
    for k in keys:
        total = total + samples[k]
    return total


def risk_statistics(losses: Sequence[float], alpha: float = 0.95) -> Dict[str, float]:
    arr = np.asarray(losses, dtype=float)
    var = float(np.quantile(arr, alpha))
    es = float(arr[arr > var].mean())
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        f"VaR_{int(alpha * 100)}": var,
        f"ES_{int(alpha * 100)}": es,
    }


def render_latex_results_table(results: Dict[str, Dict[str, float]]) -> str:
    """Render a compact LaTeX table for simulation outputs."""
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Copula & Mean & Std.\ dev. & VaR 95\% & ES 95\% \\",
        r"\hline",
    ]
    for name, stats in results.items():
        lines.append(
            f"{name} & {stats['mean']:.4f} & {stats['std']:.4f} & "
            f"{stats['VaR_95']:.4f} & {stats['ES_95']:.4f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def render_var_es_latex_table(results: Dict[str, Dict[str, float]]) -> str:
    lines = [
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Scenario & VaR 95\% & ES 95\% \\",
        r"\hline",
    ]
    for name, stats in results.items():
        lines.append(
            f"{name} & {stats['VaR_95']:.4f} & {stats['ES_95']:.4f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Example tree and demos
# ---------------------------------------------------------------------------

def make_two_level_example_tree(
        root_sampler: CopulaSampler,
        p1_sampler: CopulaSampler,
        p2_sampler: CopulaSampler,
        root_family: str,
        p1_family: str,
        p2_family: str,
        root_param: float,
        p1_param: float,
        p2_param: float,
        root_tau: Optional[float] = None,
        p1_tau: Optional[float] = None,
        p2_tau: Optional[float] = None,
) -> Node:
    return Node(
        name="root",
        copula=root_sampler,
        copula_family=root_family,
        parameter=root_param,
        kendall_tau=root_tau,
        children=[
            Node(
                name="p1",
                children=[Node("mr1"), Node("cr1")],
                copula=p1_sampler,
                copula_family=p1_family,
                parameter=p1_param,
                kendall_tau=p1_tau,
            ),
            Node(
                name="p2",
                children=[Node("mr2"), Node("cr2")],
                copula=p2_sampler,
                copula_family=p2_family,
                parameter=p2_param,
                kendall_tau=p2_tau,
            ),
        ],
    )


def make_example_samples(rng: np.random.Generator, n: int) -> Dict[str, FloatArray]:
    return {
        "root/p1/mr1": rng.lognormal(mean=2.4, sigma=0.1, size=n),
        "root/p1/cr1": rng.pareto(3.0, size=n) + 6.0,
        "root/p2/mr2": rng.lognormal(mean=2.3, sigma=0.1, size=n),
        "root/p2/cr2": rng.pareto(2.5, size=n) + 8.0,
    }


def example_gaussian() -> Dict[str, float]:
    rng = np.random.default_rng(0)
    n = 10_000

    tau_p1 = 0.54
    tau_p2 = 1.0 / 3.0
    tau_root = 1.0 / 3.0

    rho_p1 = gaussian_rho_from_kendall_tau(tau_p1)
    rho_p2 = gaussian_rho_from_kendall_tau(tau_p2)
    rho_root = gaussian_rho_from_kendall_tau(tau_root)

    tree = make_two_level_example_tree(
        root_sampler=exchangeable_gaussian_sampler(rho_root),
        p1_sampler=exchangeable_gaussian_sampler(rho_p1),
        p2_sampler=exchangeable_gaussian_sampler(rho_p2),
        root_family="Gaussian",
        p1_family="Gaussian",
        p2_family="Gaussian",
        root_param=rho_root,
        p1_param=rho_p1,
        p2_param=rho_p2,
        root_tau=tau_root,
        p1_tau=tau_p1,
        p2_tau=tau_p2,
    )

    samples = make_example_samples(rng, n)
    aligned = apply_node_aggregation_recursively(tree, samples, rng)
    losses = aggregate_by_prefix(aligned, "root/")
    return risk_statistics(losses)


def example_mirrored_clayton() -> Dict[str, float]:
    rng = np.random.default_rng(1)
    n = 10_000

    tau_p1 = 0.54
    tau_p2 = 1.0 / 3.0
    tau_root = 1.0 / 3.0

    theta_p1 = mirrored_clayton_theta_from_kendall_tau(tau_p1)
    theta_p2 = mirrored_clayton_theta_from_kendall_tau(tau_p2)
    theta_root = mirrored_clayton_theta_from_kendall_tau(tau_root)

    tree = make_two_level_example_tree(
        root_sampler=exchangeable_mirrored_clayton_sampler(theta_root),
        p1_sampler=exchangeable_mirrored_clayton_sampler(theta_p1),
        p2_sampler=exchangeable_mirrored_clayton_sampler(theta_p2),
        root_family="Mirrored Clayton",
        p1_family="Mirrored Clayton",
        p2_family="Mirrored Clayton",
        root_param=theta_root,
        p1_param=theta_p1,
        p2_param=theta_p2,
        root_tau=tau_root,
        p1_tau=tau_p1,
        p2_tau=tau_p2,
    )

    samples = make_example_samples(rng, n)
    aligned = apply_node_aggregation_recursively(tree, samples, rng)
    losses = aggregate_by_prefix(aligned, "root/")
    return risk_statistics(losses)


def run_examples() -> Dict[str, Dict[str, float]]:
    results = {
        "Gaussian": example_gaussian(),
        "Mirrored Clayton": example_mirrored_clayton(),
    }
    print(render_latex_results_table(results))
    return results


# ---------------------------------------------------------------------------
# Robustness workflow across multiple credit indices
# ---------------------------------------------------------------------------

def calibrate_and_simulate_one_credit_index(
        market_csv: str,
        credit_csv: str,
        market_column: str,
        credit_column: str,
        n_sim: int = 10000,
        base_seed: int = 42,
) -> Dict[str, object]:
    """
    Calibrate on the standard windows for one credit proxy and run the
    corresponding Gaussian and mirrored Clayton simulations.
    """
    calibration = calibrate_standard_windows_from_fred_csv(
        market_csv=market_csv,
        credit_csv=credit_csv,
        market_column=market_column,
        credit_column=credit_column,
    )

    unavailable = [r for r in calibration if r.get("kendall_tau") is None]
    if unavailable:
        raise ValueError(
            f"Calibration unavailable for {credit_column}: "
            + "; ".join(str(r.get("status", "unknown error")) for r in unavailable)
        )

    full = calibration[0]
    covid = calibration[1]

    rho_full = float(full["gaussian_rho"])
    rho_covid = float(covid["gaussian_rho"])
    theta_full = float(full["mirrored_clayton_theta"])
    theta_covid = float(covid["mirrored_clayton_theta"])

    var_es = {
        "Gaussian (Full)": simulate_from_calibrated_copula(
            rho_full, theta_full, "gaussian", n=n_sim, seed=base_seed
        ),
        "Gaussian (COVID)": simulate_from_calibrated_copula(
            rho_covid, theta_covid, "gaussian", n=n_sim, seed=base_seed + 1
        ),
        "Clayton (Full)": simulate_from_calibrated_copula(
            rho_full, theta_full, "clayton", n=n_sim, seed=base_seed + 2
        ),
        "Clayton (COVID)": simulate_from_calibrated_copula(
            rho_covid, theta_covid, "clayton", n=n_sim, seed=base_seed + 3
        ),
    }

    return {
        "credit_column": credit_column,
        "calibration": calibration,
        "var_es": var_es,
    }


def run_robustness_suite(
        market_csv: str,
        market_column: str,
        credit_specs: Dict[str, Tuple[str, str]],
        n_sim: int = 10000,
) -> Dict[str, Dict[str, object]]:
    """
    Run calibration + VaR/ES simulations for several credit proxies.

    credit_specs maps a display label to (csv_path, fred_column_name).
    """
    out: Dict[str, Dict[str, object]] = {}
    for idx, (label, (credit_csv, credit_column)) in enumerate(credit_specs.items()):
        out[label] = calibrate_and_simulate_one_credit_index(
            market_csv=market_csv,
            credit_csv=credit_csv,
            market_column=market_column,
            credit_column=credit_column,
            n_sim=n_sim,
            base_seed=100 + 10 * idx,
        )
    return out


def render_robustness_calibration_latex_table(results: Dict[str, Dict[str, object]]) -> str:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Credit proxy & Window & $N$ & $\tau$ & $\rho_{\mathrm{Gauss}}$ & $\theta_{\mathrm{Clayton}}$ \\",
        r"\hline",
    ]
    for label, payload in results.items():
        calibration = payload["calibration"]
        for row in calibration:
            lines.append(
                f"{label} & {row['window']} & {int(row['n_obs'])} & "
                f"{row['kendall_tau']:.4f} & {row['gaussian_rho']:.4f} & "
                f"{row['mirrored_clayton_theta']:.4f} \\\\"
            )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def render_robustness_var_es_latex_table(results: Dict[str, Dict[str, object]]) -> str:
    lines = [
        r"\begin{tabular}{llcc}",
        r"\hline",
        r"Credit proxy & Scenario & VaR 95\% & ES 95\% \\",
        r"\hline",
    ]
    order = ["Gaussian (Full)", "Gaussian (COVID)", "Clayton (Full)", "Clayton (COVID)"]
    for label, payload in results.items():
        stats_map = payload["var_es"]
        for scenario in order:
            stats = stats_map[scenario]
            lines.append(
                f"{label} & {scenario} & {stats['VaR_95']:.4f} & {stats['ES_95']:.4f} \\\\"
            )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


if __name__ == "__main__":

    base_dir = Path("data/")
    market_csv = str(base_dir / "SP500.csv")
    market_column = "SP500"

    credit_specs = {
        "BBB": (str(base_dir / "BAMLC0A4CBBB.csv"), "BAMLC0A4CBBB"),
        "IG": (str(base_dir / "BAMLC0A0CM.csv"), "BAMLC0A0CM"),
        "HY": (str(base_dir / "BAMLH0A0HYM2.csv"), "BAMLH0A0HYM2"),
    }

    robustness_results = run_robustness_suite(
        market_csv=market_csv,
        market_column=market_column,
        credit_specs=credit_specs,
        n_sim=10000,
    )

    for label, payload in robustness_results.items():
        print("=" * 80)
        print(f"CREDIT PROXY: {label}")
        print("CALIBRATION RESULTS:")
        print(payload["calibration"])
        print()
        print("VAR / ES RESULTS:")
        for scenario, stats in payload["var_es"].items():
            print(scenario, stats)
        print()

    print("=" * 80)
    print("LATEX TABLE (CALIBRATION ROBUSTNESS):")
    print(render_robustness_calibration_latex_table(robustness_results))
    print()
    print("LATEX TABLE (VAR/ES ROBUSTNESS):")
    print(render_robustness_var_es_latex_table(robustness_results))
