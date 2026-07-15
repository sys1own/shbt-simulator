# shbt-simulator

High-precision Rust/PyO3 simulator for the **Spacetime Holographic Bit Theory (SHBT)** boundary-to-bulk pipeline described in `main.pdf`. It computes the modular anyon boundary (`StaticBoundary`), the holographic metric projection (`HolographicProjection`), the observer causal point (`CausalPoint`), and the baryogenesis benchmark (`BaryogenesisOptimizer`).

## What is SHBT?

SHBT is a constructive toy model that derives an emergent 3+1-dimensional bulk geometry from the algebraic data of an \(SU(2)_{26} \otimes SU(3)_8 \otimes SO(10)_{312}\) anyon boundary. The simulator verifies the paper's four tables:

- **Table 1** – boundary equation verification (framing defect, modular invariance, zero-energy lock, 26→4 projection)
- **Table 2** – holographic projection audit (metric slices, positive-definiteness, trace normalization)
- **Table 3** – observer memory budget (entropy limit, past light-cone samples, property packets)
- **Table 4** – baryogenesis benchmark (\(\eta_b\), CPU-cycle reduction, stress-energy preservation)

## Dependencies

- Rust (current stable, ≥ 1.85 recommended because `az 1.3.0` is in the dependency tree)
- Python 3.x
- `m4`, `libgmp-dev`, `libmpfr-dev`, `libmpc-dev` (needed to build `rug`)
- `maturin` or `setuptools-rust` (only if you want to install the Python extension)

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y m4 libgmp-dev libmpfr-dev libmpc-dev
```

## Build

```bash
cargo build --release
```

This produces the Rust native extension at `target/release/libshbt_simulator.so`.

## Run the Rust unit tests

```bash
cargo test --release
```

All 14 tests should pass.

## Run the Python example

The fastest way is to install the extension with `maturin`:

```bash
pip install maturin
maturin develop --release
python examples/run_audit.py
```

Alternatively, after `cargo build --release`, copy the shared object so Python can import it as `shbt_simulator`:

```bash
cp target/release/libshbt_simulator.so target/release/shbt_simulator.so
PYTHONPATH=target/release python examples/run_audit.py
```

You should see the branch `(26, 8, 312)`, a framing defect near zero, `modular_invariant=True`, `zero_energy_locked=True`, `projection_dimension_26_to_4=True`, `eta_b ≈ 6.449923359416e-10`, and `stress_energy_preserved=True`.

## Python bindings

All SHBT audit structs and the top-level `ShbtSimulator` are exposed as `#[pyclass]` objects in the `shbt_simulator` module. The easiest entry point from Python is:

```python
import shbt_simulator
sim = shbt_simulator.ShbtSimulator()
report = sim.run_full_audit()
print(report.branch, report.eta_b, report.stress_energy_preserved)
```

Full paper-to-code mapping is maintained in `paper_references.md`.
