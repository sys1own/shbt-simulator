"""SHBT research simulation CLI and programmable API.

This module wraps the compiled `shbt_simulator` Rust extension and adds
data export (JSON/CSV/HDF5), optional plotting, and parameter sweeps.

Example CLI::

    python shbt_simulate.py --mode all --output result.json
    python shbt_simulate.py --mode cosmology --format csv --output slices
    python shbt_simulate.py --sweep sweep.json --output sweep.json

Programmatic usage::

    import shbt_simulate
    result = shbt_simulate.simulate({"mode": "all"})
    shbt_simulate.export_result(result, "result.json", fmt="json")
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import shbt_simulator as _rs  # type: ignore[import]
    _HAS_RUST = True
except Exception as exc:  # pragma: no cover
    _rs = None  # type: ignore[assignment]
    _HAS_RUST = False
    _IMPORT_ERROR = exc

try:
    import matplotlib  # type: ignore[import]
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt  # type: ignore[import]
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False

try:
    import h5py  # type: ignore[import]
    _HAS_H5 = True
except Exception:  # pragma: no cover
    _HAS_H5 = False

try:
    import pandas as _pd  # type: ignore[import]
    _HAS_PANDAS = True
except Exception:  # pragma: no cover
    _HAS_PANDAS = False


def _ensure_rust() -> None:
    if not _HAS_RUST:
        raise RuntimeError(
            "The compiled `shbt_simulator` Rust extension is not available. "
            "Build it first with `cargo build --release` and make sure the shared "
            "object is on your PYTHONPATH (e.g. copy target/release/libshbt_simulator.so "
            "to target/release/shbt_simulator.so)."
        ) from _IMPORT_ERROR


def _flatten_dict(prefix: str, obj: Any) -> dict[str, Any]:
    """Flatten a nested dict/list into dotted scalar columns."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_dict(key, v))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.update(_flatten_dict(key, v))
    else:
        out[prefix] = obj
    return out


def _rows_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a list of flat dicts, one per record, normalised to the union of keys."""
    flat = [_flatten_dict("", r) for r in records]
    if not flat:
        return []
    keys = sorted({k for row in flat for k in row.keys()})
    return [{k: row.get(k, "") for k in keys} for row in flat]


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_records_csv(records: list[dict[str, Any]], path: Path) -> None:
    _write_csv(_rows_from_records(records), path)


def _write_hdf5_group(group: Any, key: str, value: Any) -> None:
    if isinstance(value, dict):
        g = group.create_group(key)
        for k, v in value.items():
            _write_hdf5_group(g, str(k), v)
    elif isinstance(value, (list, tuple)):
        if not value:
            group.create_dataset(key, data=[])
        elif all(isinstance(x, (int, float, bool)) for x in value):
            group.create_dataset(key, data=value)
        elif all(isinstance(x, str) for x in value):
            dt = h5py.string_dtype()
            arr = [str(x).encode("utf-8") for x in value]
            group.create_dataset(key, data=arr, dtype=dt)
        elif all(isinstance(x, dict) for x in value):
            g = group.create_group(key)
            for i, x in enumerate(value):
                _write_hdf5_group(g, f"row_{i}", x)
        else:
            group.create_dataset(key, data=[str(x) for x in value])
    elif isinstance(value, str):
        group.create_dataset(key, data=value.encode("utf-8"))
    elif isinstance(value, (int, float, bool)):
        group.create_dataset(key, data=value)
    elif value is None:
        group.create_dataset(key, data="")
    else:
        group.create_dataset(key, data=str(value))


def _metric_slices_for_export(result: dict[str, Any]) -> list[dict[str, Any]]:
    if "cosmology" in result and result["cosmology"]:
        return result["cosmology"]
    if "audit" in result and result["audit"].get("metric_slices"):
        return result["audit"]["metric_slices"]
    return []


def _history_entries_for_export(result: dict[str, Any]) -> list[dict[str, Any]]:
    if "history" in result and result["history"]:
        return result["history"]
    if "audit" in result and result["audit"].get("history_entries"):
        return result["audit"]["history_entries"]
    return []


def export_result(
    result: dict[str, Any],
    output_path: str | Path,
    fmt: str = "json",
) -> list[Path]:
    """Export a simulation result to the requested format.

    Supported formats:
      - ``json``: single JSON file with the full result tree.
      - ``csv``: two CSV files, ``{output_path}_metric_slices.csv`` and
        ``{output_path}_history.csv``.
      - ``hdf5``: single HDF5 file with groups ``/audit``, ``/cosmology``,
        ``/history``, and ``/baryogenesis``.
    """
    output_path = Path(output_path)
    fmt = fmt.lower()

    if fmt == "json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        return [output_path]

    if fmt == "csv":
        base = output_path.with_suffix("") if output_path.suffix else output_path
        paths: list[Path] = []
        slices = _metric_slices_for_export(result)
        if slices:
            p = Path(str(base) + "_metric_slices.csv")
            _write_records_csv(slices, p)
            paths.append(p)
        history = _history_entries_for_export(result)
        if history:
            p = Path(str(base) + "_history.csv")
            _write_records_csv(history, p)
            paths.append(p)
        if not paths:
            p = Path(str(base) + "_summary.csv")
            _write_csv([_flatten_dict("", result)], p)
            paths.append(p)
        return paths

    if fmt in ("hdf5", "h5"):
        if not _HAS_H5:
            raise RuntimeError("HDF5 export requires `h5py`. Install it from requirements.txt.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as f:
            _write_hdf5_group(f, "shbt", result)
        return [output_path]

    raise ValueError(f"Unknown export format: {fmt!r}")


def _sweep_combinations(sweep_config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield dicts for every Cartesian combination of parameter lists."""
    keys = list(sweep_config.keys())
    values = [sweep_config[k] if isinstance(sweep_config[k], list) else [sweep_config[k]] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _numeric_or_none(x: Any) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _plot_result(result: dict[str, Any], prefix: Path, *, sweep: bool = False) -> list[Path]:
    """Generate optional plots and return the written file paths."""
    if not _HAS_MPL:
        logging.warning("Plotting requested but matplotlib is not installed; skipping.")
        return []

    files: list[Path] = []
    slices = _metric_slices_for_export(result)

    # 1. Metric eigenvalues vs tau/redshift
    if slices:
        fig, ax = _plt.subplots(figsize=(6, 4))
        taus = [s.get("tau", i) for i, s in enumerate(slices)]
        for dim in range(4):
            vals = [s.get("eigenvalues", [None] * 4)[dim] for s in slices]
            if all(v is not None for v in vals):
                ax.plot(taus, vals, marker="o", label=f"λ{dim}")
        ax.set_xlabel("τ")
        ax.set_ylabel("Metric eigenvalue")
        ax.set_title("Metric eigenvalues across bulk slices")
        ax.legend()
        fig.tight_layout()
        p = prefix.parent / (prefix.name + "_eigenvalues.png")
        fig.savefig(p)
        files.append(p)
        _plt.close(fig)

    # 2. Heatmap of spatial metric components from the first slice
    if slices and slices[0].get("spatial_metric"):
        fig, ax = _plt.subplots(figsize=(5, 4))
        sm = slices[0]["spatial_metric"]
        im = ax.imshow(sm, cmap="viridis", aspect="auto")
        ax.set_title("Spatial metric components (first slice)")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        p = prefix.parent / (prefix.name + "_spatial_metric.png")
        fig.savefig(p)
        files.append(p)
        _plt.close(fig)

    # 3. eta_b sweep / bar chart
    if sweep and "sweep_results" in result:
        configs = result.get("sweep_configs", [])
        results = result["sweep_results"]
        eta_values = [r.get("audit", {}).get("eta_b") or r.get("baryogenesis", {}).get("identity", {}).get("eta_b") for r in results]
        # Try to find a numeric varying parameter for the x-axis
        varying_numeric = None
        for key in configs[0].keys() if configs else []:
            vals = [_numeric_or_none(c.get(key)) for c in configs]
            if all(v is not None for v in vals):
                varying_numeric = key
                xvals = vals
                break
        fig, ax = _plt.subplots(figsize=(7, 4))
        if varying_numeric:
            ax.plot(xvals, eta_values, marker="o")
            ax.set_xlabel(varying_numeric)
        else:
            labels = [str(c.get("branch", i)) for i, c in enumerate(configs)]
            ax.bar(range(len(labels)), eta_values)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_xlabel("sweep index")
        ax.set_ylabel("η_b")
        ax.set_title("Baryon asymmetry across sweep configurations")
        fig.tight_layout()
        p = prefix.parent / (prefix.name + "_eta_b.png")
        fig.savefig(p)
        files.append(p)
        _plt.close(fig)

    return files


def simulate(config: dict[str, Any]) -> dict[str, Any]:
    """Run an SHBT simulation according to *config* and return a JSON-serialisable dict.

    Config keys:
      - mode: ``"audit"`` | ``"cosmology"`` | ``"baryogenesis"`` | ``"history"`` | ``"all"``
      - branch: tuple/list of three ints, default ``(26, 8, 312)``
      - observer_radius_fraction: float, default ``0.125``
      - redshift_max: float, default ``3.0``
      - redshift_samples: int, default ``9``
      - particles: int, default ``512``
    """
    _ensure_rust()
    assert _rs is not None

    mode = config.get("mode", "all")
    branch_raw = config.get("branch", (26, 8, 312))
    branch = tuple(int(x) for x in branch_raw)
    if len(branch) != 3:
        raise ValueError(f"branch must be a tuple/list of three integers, got {branch!r}")
    observer_radius_fraction = float(config.get("observer_radius_fraction", 0.125))
    redshift_max = float(config.get("redshift_max", 3.0))
    redshift_samples = int(config.get("redshift_samples", 9))
    particles = int(config.get("particles", 512))

    sim = _rs.ShbtSimulator.with_config(
        branch,
        observer_radius_fraction,
        redshift_max,
        redshift_samples,
        particles,
    )

    result: dict[str, Any] = {
        "config": {
            "mode": mode,
            "branch": branch,
            "observer_radius_fraction": observer_radius_fraction,
            "redshift_max": redshift_max,
            "redshift_samples": redshift_samples,
            "particles": particles,
        },
    }

    if mode in ("audit", "all"):
        report = sim.run_full_audit()
        result["audit"] = report.to_dict()

    if mode in ("cosmology", "all"):
        slices = sim.simulate_cosmology(redshift_max, redshift_samples)
        result["cosmology"] = [s.to_dict() for s in slices]

    if mode in ("baryogenesis", "all"):
        identity = sim.baryogenesis_identity()
        benchmark = sim.baryogenesis_benchmark()
        result["baryogenesis"] = {
            "identity": identity.to_dict(),
            "benchmark": benchmark.to_dict(),
        }

    if mode in ("history", "all"):
        entries = sim.crystallize_history()
        result["history"] = [e.to_dict() for e in entries]

    if mode not in ("audit", "cosmology", "baryogenesis", "history", "all"):
        raise ValueError(f"unknown simulation mode: {mode}")

    return result


def run_sweep(sweep_config: dict[str, Any]) -> dict[str, Any]:
    """Run ``simulate`` for every Cartesian product of parameter lists in *sweep_config*.

    Returns a dict with ``sweep_configs`` (the list of concrete configs) and
    ``sweep_results`` (their outputs).
    """
    configs = list(_sweep_combinations(sweep_config))
    results: list[dict[str, Any]] = []
    total = len(configs)
    for i, cfg in enumerate(configs, 1):
        logging.info("Sweep run %d/%d with config %s", i, total, cfg)
        results.append(simulate(cfg))
    return {"sweep_configs": configs, "sweep_results": results}


def _summarise(result: dict[str, Any]) -> dict[str, Any]:
    """Return a small human-readable summary of the result."""
    summary: dict[str, Any] = {"branch": result["config"]["branch"]}

    if "audit" in result:
        audit = result["audit"]
        summary["framing_defect"] = audit.get("boundary_report", {}).get("framing_defect")
        summary["modular_invariant"] = audit.get("boundary_report", {}).get("modular_invariant")
        summary["zero_energy_locked"] = audit.get("boundary_report", {}).get("zero_energy_locked")
        summary["projection_dimension_26_to_4"] = audit.get("boundary_report", {}).get("projection_dimension_26_to_4")
        summary["eta_b"] = audit.get("eta_b")
        summary["stress_energy_preserved"] = audit.get("stress_energy_preserved")
        summary["metric_slices"] = len(audit.get("metric_slices", []))
        summary["history_entries"] = len(audit.get("history_entries", []))

    if "cosmology" in result:
        summary["cosmology_slices"] = len(result["cosmology"])

    if "baryogenesis" in result:
        summary["eta_b"] = result["baryogenesis"]["identity"].get("eta_b")
        summary["stress_energy_preserved"] = result["baryogenesis"]["benchmark"].get("stress_energy_preserved")

    if "history" in result:
        summary["history_entries"] = len(result["history"])

    if "sweep_results" in result:
        summary["sweep_runs"] = len(result["sweep_results"])

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shbt_simulate.py",
        description="Run custom SHBT simulations using the Rust/PyO3 simulator.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["audit", "cosmology", "baryogenesis", "history", "all"],
        default="all",
        help="simulation mode (default: all)",
    )
    parser.add_argument(
        "--branch",
        "-b",
        nargs=3,
        type=int,
        default=(26, 8, 312),
        metavar=("K_L", "K_Q", "K"),
        help="boundary branch as three integers (default: 26 8 312)",
    )
    parser.add_argument(
        "--observer-radius-fraction",
        "-r",
        type=float,
        default=0.125,
        help="observer radius as a fraction of the global horizon (default: 0.125)",
    )
    parser.add_argument(
        "--redshift-max",
        "-z",
        type=float,
        default=3.0,
        help="maximum redshift for the past light cone (default: 3.0)",
    )
    parser.add_argument(
        "--redshift-samples",
        "-n",
        type=int,
        default=9,
        help="number of redshift / causal samples (default: 9)",
    )
    parser.add_argument(
        "--particles",
        "-p",
        type=int,
        default=512,
        help="particle count for the baryogenesis benchmark (default: 512)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="optional path to write output (extension selects format if --format is omitted)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "csv", "hdf5", "h5"],
        default=None,
        help="output format (default inferred from --output, otherwise json)",
    )
    parser.add_argument(
        "--sweep",
        "-s",
        type=str,
        default=None,
        help="path to a JSON file with parameter lists to sweep over",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="generate optional matplotlib plots alongside the data export",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="print verbose progress logging",
    )
    return parser


def _infer_format(output_path: str | None, explicit: str | None) -> str:
    if explicit:
        return explicit.lower()
    if output_path:
        ext = Path(output_path).suffix.lower()
        if ext in (".csv",):
            return "csv"
        if ext in (".h5", ".hdf5"):
            return "hdf5"
        return "json"
    return "json"


def _plot_prefix(output_path: str | None) -> Path:
    if output_path:
        p = Path(output_path)
        return p.with_name(p.stem)
    return Path("shbt")


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    fmt = _infer_format(args.output, args.format)

    if args.sweep:
        sweep_config = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
        logging.info("Running parameter sweep from %s", args.sweep)
        result = run_sweep(sweep_config)
    else:
        config = {
            "mode": args.mode,
            "branch": tuple(args.branch),
            "observer_radius_fraction": args.observer_radius_fraction,
            "redshift_max": args.redshift_max,
            "redshift_samples": args.redshift_samples,
            "particles": args.particles,
        }
        logging.info("Starting SHBT simulation: mode=%s, config=%s", args.mode, config)
        result = simulate(config)

    if args.output:
        paths = export_result(result, args.output, fmt=fmt)
        for p in paths:
            print(f"Wrote SHBT output to {p}")
        if args.plot:
            prefix = _plot_prefix(args.output)
            plot_paths = _plot_result(result, prefix, sweep=args.sweep is not None)
            for p in plot_paths:
                print(f"Wrote plot to {p}")
    else:
        if args.sweep:
            summary = {"sweep_runs": len(result["sweep_results"])}
        else:
            summary = _summarise(result)
        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
