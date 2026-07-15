"""SHBT research simulation CLI and programmable API.

Usage:
    python shbt_simulate.py --mode audit
    python shbt_simulate.py --mode cosmology --redshift-max 3.0 --redshift-samples 9
    python shbt_simulate.py --mode all --output results.json --verbose

The `simulate(config)` function can also be imported and used programmatically.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

try:
    import shbt_simulator as _rs  # type: ignore[import]
    _HAS_RUST = True
except Exception as exc:  # pragma: no cover
    _rs = None  # type: ignore[assignment]
    _HAS_RUST = False
    _IMPORT_ERROR = exc


def _ensure_rust() -> None:
    if not _HAS_RUST:
        raise RuntimeError(
            "The compiled `shbt_simulator` Rust extension is not available. "
            "Build it first with `cargo build --release` and make sure the shared "
            "object is on your PYTHONPATH (e.g. copy target/release/libshbt_simulator.so "
            "to target/release/shbt_simulator.so)."
        ) from _IMPORT_ERROR


def _bulk_metric_slice_to_dict(obj: Any) -> dict[str, Any]:
    return obj.to_dict()  # type: ignore[union-attr]


def _coordinate_log_entry_to_dict(obj: Any) -> dict[str, Any]:
    return obj.to_dict()  # type: ignore[union-attr]


def simulate(config: dict[str, Any]) -> dict[str, Any]:
    """Run an SHBT simulation according to *config* and return a JSON-serialisable dict.

    Config keys:
      - mode: "audit" | "cosmology" | "baryogenesis" | "history" | "all" (default "all")
      - branch: (int, int, int) default (26, 8, 312)
      - observer_radius_fraction: float default 0.125
      - redshift_max: float default 3.0
      - redshift_samples: int default 9
      - particles: int default 512
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
        result["cosmology"] = [_bulk_metric_slice_to_dict(s) for s in slices]

    if mode in ("baryogenesis", "all"):
        identity = sim.baryogenesis_identity()
        benchmark = sim.baryogenesis_benchmark()
        result["baryogenesis"] = {
            "identity": identity.to_dict(),
            "benchmark": benchmark.to_dict(),
        }

    if mode in ("history", "all"):
        entries = sim.crystallize_history()
        result["history"] = [_coordinate_log_entry_to_dict(e) for e in entries]

    if mode not in ("audit", "cosmology", "baryogenesis", "history", "all"):
        raise ValueError(f"unknown simulation mode: {mode}")

    return result


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
        help="optional path to write JSON output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="print verbose progress logging",
    )
    return parser


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

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

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
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"Wrote SHBT simulation result to {args.output}")
    else:
        summary = _summarise(result)
        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
