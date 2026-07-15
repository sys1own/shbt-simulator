"""Run the unified SHBT audit from Python.

This script imports the compiled `shbt_simulator` extension, builds a
`ShbtSimulator`, and prints the key benchmark results.
"""
import shbt_simulator


def main() -> None:
    sim = shbt_simulator.ShbtSimulator()
    report = sim.run_full_audit()

    print("SHBT full audit report")
    print("-" * 40)
    print("branch:", report.branch)
    print("framing defect (delta_fr):", report.framing_defect)
    print("modular_invariant:", report.modular_invariant)
    print("zero_energy_locked:", report.zero_energy_locked)
    print("projection_dimension_26_to_4:", report.projection_dimension_26_to_4)
    print("eta_b:", report.eta_b)
    print("stress_energy_preserved:", report.stress_energy_preserved)
    print("projection all passed:", report.projection_all_passed)
    print("memory all passed:", report.memory_all_passed)
    print("metric slices:", report.metric_slice_count)
    print("history entries:", report.history_entry_count)


if __name__ == "__main__":
    main()
