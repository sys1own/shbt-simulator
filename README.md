# SHBT Simulator — Static Holographic Boundary Theory

[![Rust](https://img.shields.io/badge/rust-1.80+-blue.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**SHBT Simulator** is an executable implementation of the Static Holographic Boundary Theory (SHBT), a first‑principles framework that derives gravitational closure, baryogenesis, and observer histories from a completed modular‑invariant boundary CFT. The theory is fully documented in the accompanying paper [`main.pdf`](main.pdf). This simulator serves as an **executable proof** of the theory: every claim, equation, and numerical prediction in the paper is audited by the Rust/Python code in this repository.

---

## 🧠 The Big Picture

SHBT postulates that the universe is described by a finite boundary register whose modular‑invariant pairing fixes:

- The canonical branch $(k_\ell, k_q, K) = (26, 8, 312)$ with zero framing defect.
- The holographic dark‑energy scale $\Lambda_{\rm holo} \simeq 1.09\times 10^{-52}\,\text{m}^{-2}$.
- A finite bit budget $N \simeq 3.31\times 10^{122}$.
- A holographic RG flow from boundary entropy densities to symmetric, trace‑normalised metric slices.
- A topological baryogenesis identity yielding $\eta_B \simeq 6.45\times 10^{-10}$.
- A Causal Point observer interface that crystallises histories only within a local entropy budget.

The **paper** (`main.pdf`) is the mathematical formulation. The **simulator** is the computational verification.

---

## 🚀 Paper ⇄ Simulator Relationship

The paper and simulator are developed **in lockstep**:

- Every equation in the paper is implemented in the Rust code.
- Every numerical audit table (Sections 8.1–8.4) can be reproduced by running the simulator.
- The paper explicitly references simulator objects, e.g.  
  `shbt_simulator.ShbtSimulator().run_full_audit()`,  
  `shbt_simulator.StaticBoundary`,  
  `shbt_simulator.HolographicProjection`, etc.
- The file [`paper_references.md`](paper_references.md) provides a complete mapping from paper sections to simulator methods, making it easy to navigate between the theory and its computational realisation.

> **If you change the code, you must also update the paper — and vice versa.**  
> The repository enforces this by having the paper’s numerical values directly tied to the simulator’s outputs.

---

## 📦 Repository Structure

```
.
├── Cargo.toml              # Rust project manifest
├── src/
│   ├── lib.rs              # Main Rust library (PyO3 bindings)
│   └── shbt/               # SHBT core modules
│       ├── boundary.rs      # StaticBoundary (modular data, defect, entropy)
│       ├── entropy_flow.rs  # HolographicProjection (RG flow)
│       ├── baryogenesis.rs  # BaryogenesisOptimizer (η_B, de-rendering)
│       └── causal_point.rs  # CausalPoint (observer memory, history)
├── examples/
│   └── run_audit.py        # Python script to run the full audit
├── tests/
│   └── test_shbt.rs        # Unit tests for all SHBT components
├── main.pdf                # The SHBT paper (formal theory)
├── paper_references.md     # Mapping from paper to code
└── README.md               # This file
```

---

## 🔧 Prerequisites

- **Rust** (1.80 or later) – [install via rustup](https://rustup.rs/)
- **Python** (3.8 or later) – with `pip`
- **maturin** (for building the Python module) – `pip install maturin`
- (Optional) **cargo‑test** for running unit tests

---

## 🏗️ Build & Install

### 1. Build the Rust library

```bash
cargo build --release
```

This produces a shared library in `target/release/`.

### 2. Build the Python bindings (via maturin)

```bash
maturin build --release
```

This will create a wheel in `target/wheels/`. Install it with:

```bash
pip install target/wheels/shbt_simulator-*.whl
```

You can also install directly from the local folder using:

```bash
maturin develop
```

Now you can import the module in Python:

```python
import shbt_simulator
```

---

## 🧪 Run the Full Audit

The main entry point is the Python script `examples/run_audit.py`. It constructs a `ShbtSimulator` object, runs the complete audit, and prints all key results.

```bash
python examples/run_audit.py
```

Expected output:

```
Branch: (26, 8, 312)
Δ_fr: 0.0
Modular invariant: True
Zero-energy locked: True
Projection dimension 26→4: True
η_B: 6.449923359416e-10
Stress-energy preserved: True
Metric slices: 9
Crystallized history entries: 9
```

### Run all unit tests

```bash
cargo test --release
```

All tests should pass, confirming that the simulator satisfies the algebraic constraints derived in the paper.

---

## 📊 Verifying the Paper’s Tables

The simulator’s audit reports (`boundary_report`, `projection_report`, `memory_report`, `baryogenesis_identity`, `benchmark_delta`) contain every value that appears in the paper’s tables. You can compare them directly with the printed outputs from `run_audit.py`.

- **Table 1** – Boundary closure audit → `report.boundary_report`
- **Table 2** – Holographic projection → `report.projection_report`
- **Table 3** – Causal Point memory → `report.memory_report`
- **Table 4** – Baryogenesis benchmark → `report.benchmark_delta`

The paper is written so that the reader can, at any point, refer to the code and see that the mathematics is implemented exactly.

---

## 🧭 Navigating the Code

### Core SHBT components

| Module | Structure | Paper Section |
|--------|-----------|---------------|
| `boundary.rs` | `StaticBoundary` | Sections 2–4 |
| `entropy_flow.rs` | `HolographicProjection`, `BulkMetricSlice` | Section 5 |
| `baryogenesis.rs` | `BaryogenesisOptimizer`, `BaryogenesisIdentity` | Section 6 |
| `causal_point.rs` | `CausalPoint`, `LightConeSample`, etc. | Section 7 |

### Existing (legacy) code in `lib.rs`

The original `lib.rs` already implements the low‑level components used by SHBT:
- High‑precision modular arithmetic (`rug::Float`, `rug::Complex`)
- `AnyonBraidingEngine` (SU(2), SU(3), SO(10) braid matrices)
- `TopologicalTracker` (anyon worldlines, fusion, stabiliser checks)
- `CircuitCompiler` (Solovay‑Kitaev, OpenQASM parsing)

These are reused by the new SHBT modules where appropriate.

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🖇️ Citation

If you use this repository or the SHBT framework in your research, please cite the paper:

```bibtex
@article{SHBT2026,
  author  = {Author Name},
  title   = {Static Holographic Boundary Theory},
  journal = {arXiv},
  year    = {2026},
  note    = {Available at \url{https://github.com/sys1own/shbt-simulator}}
}
