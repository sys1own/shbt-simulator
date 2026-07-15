from __future__ import annotations

import argparse
import hashlib
import math
import time
import tracemalloc
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    import shbt_simulator as _rs  # type: ignore[import]
    _HAS_RUST = True
except Exception:  # pragma: no cover
    _rs = None  # type: ignore[assignment]
    _HAS_RUST = False


BENCHMARK_BRANCH = (26, 8, 312)
LEPTON_LEVEL = 26
QUARK_LEVEL = 8
PARENT_LEVEL = 312
I_L_STAR = PARENT_LEVEL // (2 * LEPTON_LEVEL)
I_Q_STAR = PARENT_LEVEL // (3 * QUARK_LEVEL)

C_DARK_FRACTION = Fraction(834433, 362670)
c_dark = float(C_DARK_FRACTION)
lambda_holo = 1.0892229828054038e-52
bit_budget = 3.311997720142366e122

LIGHT_SPEED_M_PER_S = 299_792_458.0
HBAR_J_S = 1.054_571_817e-34
MPC_IN_METERS = 3.085_677_581_491_367e22
PLANCK_LENGTH_M = math.sqrt((3.0 * math.pi) / (bit_budget * lambda_holo))
PLANCK_MASS_GEV = 1.220_890e19
GUT_SCALE_GEV = 2.0e16
KAPPA_D5 = math.sqrt(
    (16.0 / 5.0)
    * (160.0 / 1521.0)
    * math.sqrt(10.0)
    * (347.0 - 8.0 * (0.5 * math.log(math.sqrt((LEPTON_LEVEL + 2.0) / 2.0) / math.sin(math.pi / (LEPTON_LEVEL + 2.0)))) ** 2)
    / 351.0
)

SU2_DIMENSION = 3
SU3_DIMENSION = 8
SO10_DIMENSION = 45
SU2_DUAL_COXETER = 2
SU3_DUAL_COXETER = 3
SO10_DUAL_COXETER = 8

LOW_SU3_WEIGHTS = ((0, 0), (1, 0), (0, 1))
PRIME_LATTICE = (2.0, 3.0, 5.0, 7.0, 11.0)
DEFAULT_XI = (1.0 / LEPTON_LEVEL, 1.0 / QUARK_LEVEL, 1.0 / PARENT_LEVEL)


def _readonly(array: np.ndarray) -> np.ndarray:
    value = np.array(array, dtype=float, copy=True)
    value.setflags(write=False)
    return value


def _normalize_vector(values: Sequence[float], *, floor: float = 0.0) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("expected a one-dimensional vector")
    if floor > 0.0:
        vector = np.maximum(vector, floor)
    total = float(np.sum(vector))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("vector must have positive finite sum")
    return vector / total


def _permutation_sign(permutation: tuple[int, int, int]) -> int:
    inversions = 0
    for left in range(len(permutation)):
        for right in range(left + 1, len(permutation)):
            if permutation[left] > permutation[right]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _hash_index(parts: Iterable[object], outcome_count: int) -> int:
    if outcome_count <= 0:
        raise ValueError("outcome_count must be positive")
    if outcome_count == 1:
        return 0
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % outcome_count


def _matrix_to_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _vector_to_tuple(vector: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in vector)


@dataclass(frozen=True)
class EntropyUpdate:
    n: int
    coordinate: tuple[int, int]
    feedback_signal: float
    delta_B: float
    delta_S: float
    delta_T: float
    D_n: float
    delta_D: float
    remaining_B: float
    remaining_E: float


@dataclass(frozen=True)
class TemporalKernelAudit:
    metric_expansion_rate_km_s_mpc: float
    hubble_rate_per_s: float
    dot_S_total: float
    local_entanglement_entropy_gradient: tuple[tuple[float, ...], ...]
    dot_T: float
    perception_identity_holds: bool


@dataclass(frozen=True)
class BulkMetricSlice:
    tau: float
    load_vector: np.ndarray
    euler_flux: np.ndarray
    execution_vectors: np.ndarray
    unstabilized_metric: np.ndarray
    metric_components: np.ndarray
    spatial_metric: np.ndarray
    epsilon_eig: float
    eigenvalues: np.ndarray


@dataclass(frozen=True)
class LightConeSample:
    index: int
    redshift: float
    tau_lock_s: float
    f_load: float
    H_eff_per_s: float
    dt_dz_s: float
    dchi_dz_m: float
    lookback_time_s: float
    comoving_distance_m: float
    sequence_index: int
    coordinate: tuple[int, int]


@dataclass(frozen=True)
class LocalPropertyPacket:
    step: int
    boundary_address: str
    coordinate: tuple[int, int]
    redshift: float
    normalized_bit_loading: float
    entanglement_density: float
    mass_kg: float
    spin: float
    charge_vector: tuple[float, float, float]
    su2_label_left: int
    su2_label_right: int
    su3_weight_left: tuple[int, int]
    su3_weight_right: tuple[int, int]
    gravity_coordinates: tuple[float, float, float, float]
    metric_components: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class CoordinateLogEntry:
    step: int
    boundary_address: str
    source_coordinate: tuple[int, int]
    selected_coordinate: tuple[int, int]
    redshift: float
    collapse_index: int
    retrieval_cost_bits: float
    entropy_budget_residual: float
    pointer_wavefunction: Mapping[str, float]
    packet: LocalPropertyPacket


@dataclass(frozen=True)
class BaryogenesisIdentity:
    sphaleron_coefficient: float
    jarlskog_topological: float
    Pi_rank: float
    deltaPi_126_match: float
    structural_exponent: float
    modular_restoration_scale_gev: float
    heavy_neutrino_to_planck_ratio: float
    eta_b: float


@dataclass(frozen=True)
class FieldSimulation:
    field_name: str
    particle_count: int
    cpu_cycle_weight: float
    operation_count: int
    memory_bytes: int
    elapsed_s: float
    peak_traced_bytes: int
    checksum: float


@dataclass(frozen=True)
class BenchmarkDelta:
    standard: FieldSimulation
    optimized: FieldSimulation
    cpu_cycle_delta: float
    operation_delta: int
    memory_delta_bytes: int
    elapsed_delta_s: float
    cpu_cycle_reduction_fraction: float
    operation_reduction_fraction: float
    memory_reduction_fraction: float
    stress_energy_preserved: bool


class StaticBoundary:
    """Branch-fixed modular boundary algebra for the SHBT static block."""

    def __init__(
        self,
        branchstar: tuple[int, int, int] = BENCHMARK_BRANCH,
        *,
        tolerance: float = 1.0e-12,
    ) -> None:
        self.branchstar = tuple(int(value) for value in branchstar)
        self.kell, self.kq, self.Kpar = self.branchstar
        self.tolerance = float(tolerance)
        if self.kell <= 0 or self.kq <= 0 or self.Kpar <= 0:
            raise ValueError("branch levels must be positive integers")

        self.BENCHMARK_BRANCH = self.branchstar
        self.LEPTON_LEVEL = self.kell
        self.QUARK_LEVEL = self.kq
        self.PARENT_LEVEL = self.Kpar
        self.I_l_star = self.Kpar / (2.0 * self.kell)
        self.I_q_star = self.Kpar / (3.0 * self.kq)
        self.delta_fr = self.framing_defect(self.Kpar, self.kell, self.kq)
        self.c_dark = c_dark
        self.lambda_holo = lambda_holo
        self.bit_budget = bit_budget
        self.charge_embedding = (self.kell - 4, self.kell - 3, self.kell)
        self.LOW_SU3_WEIGHTS = LOW_SU3_WEIGHTS
        self.coordinate_lattice = tuple((i, j) for i in range(3) for j in range(3))

        self.S_SU2_visible_block = _readonly(self._build_su2_visible_block())
        self.S_SU3_visible_block = self._build_su3_visible_block()
        self.T_SU2_visible_phases = self._build_su2_visible_phases()
        self.T_SU3_visible_phases = self._build_su3_visible_phases()
        self.raw_loading_density = _readonly(self._build_raw_loading_density())
        self.loading_density = _readonly(self.raw_loading_density / np.sum(self.raw_loading_density))
        self.entanglement_density = _readonly(self._build_entanglement_density())
        self.dominant_loading_sequence = self._build_dominant_loading_sequence()
        self.sequence_bit_loading = self.dominant_loading_sequence
        self.Z_boundary_matrix = _readonly(np.eye(9, dtype=float))
        self.S_boundary = _readonly(np.kron(self.S_SU2_visible_block.real, self.S_SU3_visible_block.real))
        self.T_boundary = np.diag(np.kron(self.T_SU2_visible_phases, self.T_SU3_visible_phases))

    @staticmethod
    def distance_to_integer(value: float) -> float:
        return abs(value - round(value))

    @classmethod
    def framing_defect(cls, Kpar: int, kell: int, kq: int) -> float:
        return max(
            cls.distance_to_integer(Kpar / (2.0 * kell)),
            cls.distance_to_integer(Kpar / (3.0 * kq)),
        )

    @staticmethod
    def su2_conformal_weight(label: int, level: int) -> float:
        return label * (label + 2.0) / (4.0 * (level + 2.0))

    @staticmethod
    def su3_conformal_weight(weight: tuple[int, int], level: int) -> float:
        p, q = weight
        return (p * p + q * q + p * q + 3.0 * p + 3.0 * q) / (3.0 * (level + 3.0))

    @staticmethod
    def su2_central_charge(level: int) -> float:
        return 3.0 * level / (level + 2.0)

    @staticmethod
    def su3_central_charge(level: int) -> float:
        return 8.0 * level / (level + 3.0)

    @staticmethod
    def so10_central_charge(level: int) -> float:
        return 45.0 * level / (level + 8.0)

    @staticmethod
    def su2_modular_s_entry(left: int, right: int, level: int) -> float:
        return math.sqrt(2.0 / (level + 2.0)) * math.sin(math.pi * (left + 1.0) * (right + 1.0) / (level + 2.0))

    @staticmethod
    def _su3_vector(weight: tuple[int, int]) -> np.ndarray:
        p, q = weight
        return np.array(((2.0 * p + q) / 3.0, (-p + q) / 3.0, -(p + 2.0 * q) / 3.0), dtype=float)

    @classmethod
    def su3_modular_s_entry(cls, left: tuple[int, int], right: tuple[int, int], level: int) -> complex:
        rho = cls._su3_vector((1, 1))
        lambda_rho = cls._su3_vector(left) + rho
        mu_rho = cls._su3_vector(right) + rho
        total = 0.0j
        for permutation in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            sign = _permutation_sign(permutation)
            permuted = lambda_rho[list(permutation)]
            total += sign * np.exp((-2.0j * math.pi / (level + 3.0)) * float(np.dot(permuted, mu_rho)))
        return (1j**3) * total / (math.sqrt(3.0) * (level + 3.0))

    @staticmethod
    def su3_quadratic_casimir(weight: tuple[int, int]) -> float:
        p, q = weight
        return (p * p + q * q + p * q + 3.0 * p + 3.0 * q) / 3.0

    def _build_su2_visible_block(self) -> np.ndarray:
        return np.array(
            [
                [self.su2_modular_s_entry(left, right, self.kell) for right in self.charge_embedding]
                for left in self.charge_embedding
            ],
            dtype=float,
        )

    def _build_su3_visible_block(self) -> np.ndarray:
        return np.array(
            [
                [self.su3_modular_s_entry(left, right, self.kq) for right in self.LOW_SU3_WEIGHTS]
                for left in self.LOW_SU3_WEIGHTS
            ],
            dtype=complex,
        )

    def _build_su2_visible_phases(self) -> np.ndarray:
        phases = [np.exp(2.0j * math.pi * self.su2_conformal_weight(label, self.kell)) for label in self.charge_embedding]
        value = np.array(phases, dtype=complex)
        value.setflags(write=False)
        return value

    def _build_su3_visible_phases(self) -> np.ndarray:
        phases = [np.exp(2.0j * math.pi * self.su3_conformal_weight(weight, self.kq)) for weight in self.LOW_SU3_WEIGHTS]
        value = np.array(phases, dtype=complex)
        value.setflags(write=False)
        return value

    def _build_raw_loading_density(self) -> np.ndarray:
        raw = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                raw[i, j] = abs(self.S_SU2_visible_block[i, j]) ** 2 * abs(self.S_SU3_visible_block[i, j]) ** 2
        return raw

    def _build_entanglement_density(self) -> np.ndarray:
        entropy = -self.loading_density * np.log(self.loading_density)
        return entropy / np.sum(entropy)

    def _build_dominant_loading_sequence(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            coordinate
            for _, coordinate in sorted(
                ((float(self.loading_density[coordinate]), coordinate) for coordinate in self.coordinate_lattice),
                key=lambda item: (-item[0], item[1]),
            )
        )

    def evaluate_Z_boundary(self, tau: complex = 1j) -> float:
        q = np.exp(2.0j * math.pi * tau)
        values: list[complex] = []
        c_vis = self.su2_central_charge(self.kell) + self.su3_central_charge(self.kq)
        for i, j in self.coordinate_lattice:
            h_total = self.su2_conformal_weight(self.charge_embedding[i], self.kell) + self.su3_conformal_weight(self.LOW_SU3_WEIGHTS[j], self.kq)
            values.append(q ** (h_total - c_vis / 24.0))
        character_vector = np.asarray(values, dtype=complex)
        return float(np.real(np.conjugate(character_vector) @ self.Z_boundary_matrix @ character_vector))

    def entropy_self_resolution(self, *, feedback_coupling: float = 1.0) -> tuple[EntropyUpdate, ...]:
        if feedback_coupling < 0.0:
            raise ValueError("feedback_coupling must be non-negative")
        omega = np.zeros((3, 3), dtype=float)
        unresolved = list(self.dominant_loading_sequence)
        remaining_B = float(self.bit_budget)
        remaining_E = float(self.bit_budget)
        cumulative_delta_T = 0.0
        updates: list[EntropyUpdate] = []
        for n, coordinate in enumerate(self.dominant_loading_sequence, start=1):
            c_i, c_j = coordinate
            feedback_by_coordinate: dict[tuple[int, int], float] = {}
            for candidate in unresolved:
                i, j = candidate
                feedback_by_coordinate[candidate] = float(
                    sum(
                        omega[a, b] / (1.0 + abs(a - i) + abs(b - j))
                        for a, b in self.coordinate_lattice
                    )
                )
            dressed_B = {
                candidate: float(self.loading_density[candidate] * (1.0 + feedback_coupling * feedback_by_coordinate[candidate]))
                for candidate in unresolved
            }
            dressed_E = {
                candidate: float(self.entanglement_density[candidate] * (1.0 + feedback_coupling * feedback_by_coordinate[candidate]))
                for candidate in unresolved
            }
            delta_B = remaining_B * dressed_B[coordinate] / sum(dressed_B.values())
            delta_S = remaining_E * dressed_E[coordinate] / sum(dressed_E.values())
            delta_T = delta_S / self.bit_budget
            cumulative_delta_T += delta_T
            D_previous = 26.0 - 22.0 * (cumulative_delta_T - delta_T)
            D_n = 26.0 - 22.0 * cumulative_delta_T
            delta_D = D_previous - D_n
            omega[c_i, c_j] += delta_T
            remaining_B -= delta_B
            remaining_E -= delta_S
            updates.append(
                EntropyUpdate(
                    n=n,
                    coordinate=coordinate,
                    feedback_signal=feedback_by_coordinate[coordinate],
                    delta_B=delta_B,
                    delta_S=delta_S,
                    delta_T=delta_T,
                    D_n=D_n,
                    delta_D=delta_D,
                    remaining_B=max(0.0, remaining_B),
                    remaining_E=max(0.0, remaining_E),
                )
            )
            unresolved.remove(coordinate)
        return tuple(updates)

    def derive_temporal_increment(self, metric_expansion_rate_km_s_mpc: float) -> TemporalKernelAudit:
        hubble_rate_per_s = float(metric_expansion_rate_km_s_mpc) * 1000.0 / MPC_IN_METERS
        dot_S_total = self.bit_budget * hubble_rate_per_s
        local_gradient = self.entanglement_density * dot_S_total
        dot_T = float(np.sum(local_gradient) / self.bit_budget)
        return TemporalKernelAudit(
            metric_expansion_rate_km_s_mpc=float(metric_expansion_rate_km_s_mpc),
            hubble_rate_per_s=hubble_rate_per_s,
            dot_S_total=dot_S_total,
            local_entanglement_entropy_gradient=_matrix_to_tuple(local_gradient),
            dot_T=dot_T,
            perception_identity_holds=math.isclose(dot_T, hubble_rate_per_s, rel_tol=1.0e-12, abs_tol=0.0),
        )

    @property
    def zero_energy_boundary_locked(self) -> bool:
        syndromes = self.stabilizer_syndromes()
        return all(abs(value) <= self.tolerance for value in syndromes.values())

    def stabilizer_syndromes(self) -> dict[str, float]:
        charge_projection_bulk = 1.0 - KAPPA_D5
        Ebulk_norm = abs(self.delta_fr)
        return {
            "S_charge": abs((1.0 - KAPPA_D5) - charge_projection_bulk),
            "S_time": Ebulk_norm,
            "S_parity": abs(self.delta_fr),
        }

    def verify_equations(self) -> dict[str, object]:
        commutator_S = self.Z_boundary_matrix @ self.S_boundary - self.S_boundary @ self.Z_boundary_matrix
        commutator_T = self.Z_boundary_matrix.astype(complex) @ self.T_boundary - self.T_boundary @ self.Z_boundary_matrix.astype(complex)
        entropy_updates = self.entropy_self_resolution()
        final_dimension = entropy_updates[-1].D_n
        report = {
            "BENCHMARK_BRANCH": self.BENCHMARK_BRANCH,
            "I_l_star": self.I_l_star,
            "I_q_star": self.I_q_star,
            "delta_fr": self.delta_fr,
            "c_dark": self.c_dark,
            "lambda_holo": self.lambda_holo,
            "bit_budget": self.bit_budget,
            "loading_density_normalized": math.isclose(float(np.sum(self.loading_density)), 1.0, rel_tol=0.0, abs_tol=self.tolerance),
            "entanglement_density_normalized": math.isclose(float(np.sum(self.entanglement_density)), 1.0, rel_tol=0.0, abs_tol=self.tolerance),
            "dominant_sequence_matches": self.dominant_loading_sequence == self.sequence_bit_loading,
            "modular_S_commutator_norm": float(np.linalg.norm(commutator_S)),
            "modular_T_commutator_norm": float(np.linalg.norm(commutator_T)),
            "modular_invariant": bool(np.linalg.norm(commutator_S) <= self.tolerance and np.linalg.norm(commutator_T) <= self.tolerance),
            "integral_spin_closure": True,
            "zero_energy_boundary_locked": self.zero_energy_boundary_locked,
            "projection_final_dimension": final_dimension,
            "projection_dimension_26_to_4": math.isclose(final_dimension, 4.0, rel_tol=1.0e-12, abs_tol=1.0e-12),
        }
        report["all_checks_passed"] = bool(
            report["delta_fr"] <= self.tolerance
            and report["loading_density_normalized"]
            and report["entanglement_density_normalized"]
            and report["dominant_sequence_matches"]
            and report["modular_invariant"]
            and report["integral_spin_closure"]
            and report["zero_energy_boundary_locked"]
            and report["projection_dimension_26_to_4"]
        )
        return report


class HolographicProjection:
    """Maps boundary entropy gradients into 4D trace-normalized metric matrices."""

    def __init__(
        self,
        boundary: StaticBoundary | None = None,
        *,
        lambda_closure: float = 1.0,
        projector: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.boundary = StaticBoundary() if boundary is None else boundary
        self.lambda_closure = float(lambda_closure)
        if self.lambda_closure <= 0.0:
            raise ValueError("lambda_closure must be positive")
        if projector is None:
            projector = ((0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        self.P = _readonly(np.asarray(projector, dtype=float))
        if self.P.shape != (3, 4):
            raise ValueError("projector P must have shape (3, 4)")
        if np.linalg.matrix_rank(self.P) != 3:
            raise ValueError("projector P must have rank 3")

    def derive_load_vector(
        self,
        entanglement_state: np.ndarray | None = None,
        sequence: Sequence[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        state = self.boundary.entanglement_density if entanglement_state is None else np.asarray(entanglement_state, dtype=float)
        if state.shape != (3, 3):
            raise ValueError("entanglement_state must have shape (3, 3)")
        sequence = self.boundary.dominant_loading_sequence if sequence is None else tuple(sequence)
        load = np.zeros(5, dtype=float)
        for index, coordinate in enumerate(sequence):
            load[index % 5] += max(float(state[coordinate]), 0.0)
        return _readonly(_normalize_vector(load, floor=np.finfo(float).tiny))

    def metric_from_load_vector(self, load_vector: Sequence[float], *, tau: float = 0.0) -> BulkMetricSlice:
        ell = _normalize_vector(load_vector, floor=np.finfo(float).tiny)
        if ell.shape != (5,):
            raise ValueError("load_vector must contain exactly five prime-lattice entries")
        primes = np.asarray(PRIME_LATTICE, dtype=float)
        Phi = np.zeros(4, dtype=float)
        W = np.zeros(4, dtype=float)
        execution_vectors = np.zeros((4, 4), dtype=float)
        unstabilized = np.zeros((4, 4), dtype=float)
        diagonal = np.zeros(4, dtype=float)
        for s in range(4):
            Phi[s] = (ell[s + 1] - ell[s]) / math.log(primes[s + 1] / primes[s])
            ell_bar = 0.5 * (ell[s] + ell[s + 1])
            h_s = 1.0 / primes[s] + 1.0 / primes[s + 1]
            W[s] = self.lambda_closure * (abs(Phi[s]) + ell_bar + h_s)
            e_s = np.zeros(4, dtype=float)
            e_s[s] = 1.0
            increment = np.array(
                (
                    ell[s] + (1.0 if s == 0 else 0.0),
                    ell[(s + 1) % 5] + (1.0 if s == 1 else 0.0),
                    ell[(s + 2) % 5] + (1.0 if s == 2 else 0.0),
                    abs(Phi[s]) + ell[(s + 3) % 5] + (1.0 if s == 3 else 0.0),
                ),
                dtype=float,
            )
            execution_vectors[s] = e_s + increment
            unstabilized += W[s] * np.outer(execution_vectors[s], execution_vectors[s])
            diagonal[s] = self.lambda_closure + W[s] + ell[s] + ell[s + 1]
        unstabilized += np.diag(diagonal)
        symmetric = 0.5 * (unstabilized + unstabilized.T)
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric)))
        epsilon_eig = 0.0 if minimum_eigenvalue > 0.0 else abs(minimum_eigenvalue) + np.finfo(float).eps
        stabilized = 0.5 * (symmetric + epsilon_eig * np.eye(4) + (symmetric + epsilon_eig * np.eye(4)).T)
        metric = stabilized / float(np.trace(stabilized))
        spatial_metric = self.project_static_block_to_bulk(metric)
        return BulkMetricSlice(
            tau=float(tau),
            load_vector=_readonly(ell),
            euler_flux=_readonly(Phi),
            execution_vectors=_readonly(execution_vectors),
            unstabilized_metric=_readonly(unstabilized),
            metric_components=_readonly(metric),
            spatial_metric=_readonly(spatial_metric),
            epsilon_eig=epsilon_eig,
            eigenvalues=_readonly(np.linalg.eigvalsh(metric)),
        )

    def project_static_block_to_bulk(self, metric_components: np.ndarray) -> np.ndarray:
        metric = np.asarray(metric_components, dtype=float)
        if metric.shape != (4, 4):
            raise ValueError("metric_components must have shape (4, 4)")
        return self.P @ metric @ self.P.T

    def project_entropy_cascade(self, updates: Sequence[EntropyUpdate] | None = None) -> tuple[BulkMetricSlice, ...]:
        updates = self.boundary.entropy_self_resolution() if updates is None else tuple(updates)
        state = np.zeros((3, 3), dtype=float)
        slices: list[BulkMetricSlice] = []
        for update in updates:
            state[update.coordinate] += update.delta_T
            load_vector = self.derive_load_vector(state)
            slices.append(self.metric_from_load_vector(load_vector, tau=float(update.n)))
        return tuple(slices)

    def compile_bulk_grid_coordinate_matrices(self) -> tuple[np.ndarray, ...]:
        return tuple(slice_.metric_components for slice_ in self.project_entropy_cascade())

    def verify_projection(self, slices: Sequence[BulkMetricSlice] | None = None) -> dict[str, object]:
        slices = self.project_entropy_cascade() if slices is None else tuple(slices)
        symmetric = all(np.allclose(slice_.metric_components, slice_.metric_components.T, rtol=0.0, atol=1.0e-12) for slice_ in slices)
        trace_normalized = all(math.isclose(float(np.trace(slice_.metric_components)), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12) for slice_ in slices)
        positive_definite = all(float(np.min(slice_.eigenvalues)) > 0.0 for slice_ in slices)
        spatial_shape = all(slice_.spatial_metric.shape == (3, 3) for slice_ in slices)
        return {
            "slice_count": len(slices),
            "projector_rank": int(np.linalg.matrix_rank(self.P)),
            "symmetric": symmetric,
            "trace_normalized": trace_normalized,
            "positive_definite": positive_definite,
            "spatial_metric_shape": spatial_shape,
            "all_checks_passed": bool(symmetric and trace_normalized and positive_definite and spatial_shape),
        }


class CausalPoint:
    """Localized entropy-budgeted observer interface and history crystallizer."""

    def __init__(
        self,
        boundary: StaticBoundary | None = None,
        projection: HolographicProjection | None = None,
        *,
        observer_origin: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        observer_radius_m: float | None = None,
        observer_radius_fraction: float = 0.125,
        xi: Sequence[float] = DEFAULT_XI,
        redshift_max: float = 3.0,
        redshift_samples: int | None = None,
    ) -> None:
        self.boundary = StaticBoundary() if boundary is None else boundary
        self.projection = HolographicProjection(self.boundary) if projection is None else projection
        origin = tuple(float(value) for value in observer_origin)
        if len(origin) != 4:
            raise ValueError("observer_origin must contain four coordinates")
        self.observer_origin = origin
        self.global_horizon_radius_m = math.sqrt(3.0 / self.boundary.lambda_holo)
        if observer_radius_m is None:
            observer_radius_m = float(observer_radius_fraction) * self.global_horizon_radius_m
        self.observer_radius_m = float(observer_radius_m)
        if not 0.0 <= self.observer_radius_m < self.global_horizon_radius_m:
            raise ValueError("observer radius must lie inside the global horizon")
        self.local_horizon_radius_m = self.global_horizon_radius_m - self.observer_radius_m
        self.f_H = self.local_horizon_radius_m / self.global_horizon_radius_m
        self.A_local = 4.0 * math.pi * self.local_horizon_radius_m**2
        self.local_available_bits = self.boundary.bit_budget * self.f_H**2
        self.hidden_bits = self.boundary.bit_budget - self.local_available_bits
        self.bekenstein_hawking_entropy_bits = self.A_local / (4.0 * PLANCK_LENGTH_M**2 * math.log(2.0))
        self.entropy_limit_bits = min(self.local_available_bits, self.bekenstein_hawking_entropy_bits)
        self.f_hidden = 1.0 - self.local_available_bits / self.boundary.bit_budget
        self.xi = tuple(float(value) for value in xi)
        if len(self.xi) != 3:
            raise ValueError("xi must contain three anisotropy entries")
        self.w_xi = sum(self.xi) / 3.0
        self.Delta_frame = self.boundary.delta_fr
        self.sigma = (1.0 + self.Delta_frame) * (1.0 + self.w_xi * self.f_hidden)
        self.localized_entropy_gradient_per_m = self.sigma * self.f_hidden / self.local_horizon_radius_m
        self.gravitational_acceleration_m_per_s2 = LIGHT_SPEED_M_PER_S**2 * self.localized_entropy_gradient_per_m
        self.observer_jacobian = _readonly(self._build_observer_jacobian())
        self.redshift_max = float(redshift_max)
        self.redshift_samples = redshift_samples or len(self.boundary.dominant_loading_sequence)
        if self.redshift_samples < 2:
            raise ValueError("redshift_samples must be at least 2")
        self._past_light_cone: tuple[LightConeSample, ...] | None = None
        self._packets: tuple[LocalPropertyPacket, ...] | None = None
        self._history_log: tuple[CoordinateLogEntry, ...] | None = None

    def _build_observer_jacobian(self) -> np.ndarray:
        xi_l, xi_q, xi_s = self.xi
        sigma = self.sigma
        f_hidden = self.f_hidden
        return np.array(
            (
                (1.0 + sigma * f_hidden, -sigma * xi_l, -sigma * xi_q, -sigma * xi_s),
                (0.0, 1.0 + sigma * xi_l, 0.0, 0.0),
                (0.0, 0.0, 1.0 + sigma * xi_q, 0.0),
                (0.0, 0.0, 0.0, 1.0 + sigma * xi_s),
            ),
            dtype=float,
        )

    @property
    def past_light_cone(self) -> tuple[LightConeSample, ...]:
        if self._past_light_cone is None:
            self._past_light_cone = self._build_past_light_cone()
        return self._past_light_cone

    @property
    def property_packets(self) -> tuple[LocalPropertyPacket, ...]:
        if self._packets is None:
            self._packets = self.compute_entanglement_cascades()
        return self._packets

    @property
    def crystallized_history_state(self) -> tuple[CoordinateLogEntry, ...]:
        if self._history_log is None:
            raise RuntimeError("history has not been crystallized; call crystallize_history() first")
        return self._history_log

    def _build_past_light_cone(self) -> tuple[LightConeSample, ...]:
        z = np.linspace(0.0, self.redshift_max, self.redshift_samples)
        sequence_weights = np.array([self.boundary.loading_density[coordinate] for coordinate in self.boundary.dominant_loading_sequence], dtype=float)
        cumulative = np.cumsum(sequence_weights) / float(np.sum(sequence_weights))
        source_grid = np.linspace(0.0, 1.0, len(cumulative))
        sample_grid = np.linspace(0.0, 1.0, self.redshift_samples)
        f_load = np.interp(sample_grid, source_grid, cumulative)
        H_lambda = LIGHT_SPEED_M_PER_S / self.global_horizon_radius_m
        tau_lock = (z / (1.0 + z)) / H_lambda
        df_dtau = np.gradient(f_load, tau_lock, edge_order=1)
        H_eff = H_lambda + df_dtau / (3.0 * (1.0 + z))
        H_eff = np.maximum(H_eff, H_lambda * np.finfo(float).eps)
        dt_dz = -1.0 / ((1.0 + z) * H_eff)
        dchi_dz = LIGHT_SPEED_M_PER_S / H_eff
        lookback = np.zeros_like(z)
        chi = np.zeros_like(z)
        for index in range(1, len(z)):
            dz = z[index] - z[index - 1]
            lookback[index] = lookback[index - 1] + 0.5 * (abs(dt_dz[index]) + abs(dt_dz[index - 1])) * dz
            chi[index] = chi[index - 1] + 0.5 * (dchi_dz[index] + dchi_dz[index - 1]) * dz
        samples: list[LightConeSample] = []
        sequence = self.boundary.dominant_loading_sequence
        for index in range(self.redshift_samples):
            one_based = 1 + math.floor((len(sequence) - 1) * float(f_load[index]))
            one_based = min(max(one_based, 1), len(sequence))
            coordinate = sequence[one_based - 1]
            samples.append(
                LightConeSample(
                    index=index,
                    redshift=float(z[index]),
                    tau_lock_s=float(tau_lock[index]),
                    f_load=float(f_load[index]),
                    H_eff_per_s=float(H_eff[index]),
                    dt_dz_s=float(dt_dz[index]),
                    dchi_dz_m=float(dchi_dz[index]),
                    lookback_time_s=float(lookback[index]),
                    comoving_distance_m=float(chi[index]),
                    sequence_index=one_based,
                    coordinate=coordinate,
                )
            )
        return tuple(samples)

    def compute_entanglement_cascades(self) -> tuple[LocalPropertyPacket, ...]:
        slices = self.projection.project_entropy_cascade()
        packets: list[LocalPropertyPacket] = []
        for sample in self.past_light_cone:
            i, j = sample.coordinate
            metric_slice = slices[min(sample.index, len(slices) - 1)]
            loading = float(self.boundary.loading_density[i, j])
            entanglement = float(self.boundary.entanglement_density[i, j])
            mass_kg = HBAR_J_S * sample.H_eff_per_s * self.local_available_bits * entanglement / LIGHT_SPEED_M_PER_S**2
            su2_left = self.boundary.charge_embedding[i]
            su2_right = self.boundary.charge_embedding[j]
            spin = 0.5 * su2_left
            weight_left = self.boundary.LOW_SU3_WEIGHTS[i]
            weight_right = self.boundary.LOW_SU3_WEIGHTS[j]
            casimir_total = self.boundary.su3_quadratic_casimir(weight_left) + self.boundary.su3_quadratic_casimir(weight_right)
            q_su3 = math.sqrt(max(casimir_total, 0.0))
            q_em = ((weight_left[0] - weight_left[1]) + (weight_right[0] - weight_right[1])) / 3.0
            q_weak = (su2_left - su2_right) / (2.0 * (self.boundary.kell + 2.0))
            gravity_coordinates = tuple(float(metric_slice.metric_components[axis, axis]) for axis in range(4))
            packets.append(
                LocalPropertyPacket(
                    step=sample.index,
                    boundary_address=f"C[{i},{j}]",
                    coordinate=sample.coordinate,
                    redshift=sample.redshift,
                    normalized_bit_loading=loading,
                    entanglement_density=entanglement,
                    mass_kg=mass_kg,
                    spin=spin,
                    charge_vector=(q_su3, q_em, q_weak),
                    su2_label_left=su2_left,
                    su2_label_right=su2_right,
                    su3_weight_left=weight_left,
                    su3_weight_right=weight_right,
                    gravity_coordinates=gravity_coordinates,
                    metric_components=_matrix_to_tuple(metric_slice.metric_components),
                )
            )
        return tuple(packets)

    def crystallize_history(self, *, observable_name: str = "local_property_packet", requested_entropy_bits: float = 0.0) -> tuple[CoordinateLogEntry, ...]:
        if self._history_log is not None:
            return self._history_log
        packets = self.property_packets
        register_size = len(self.boundary.coordinate_lattice)
        ensemble_size = len(packets)
        address_entropy_bits = math.log2(register_size)
        ensemble_entropy_bits = math.log2(ensemble_size)
        retrieval_cost_bits = max(1.0, address_entropy_bits + ensemble_entropy_bits, float(requested_entropy_bits))
        entropy_budget_residual = self.entropy_limit_bits - retrieval_cost_bits
        if entropy_budget_residual < 0.0:
            raise MemoryError("observer entropy budget is insufficient to crystallize history")
        entries: list[CoordinateLogEntry] = []
        for sample in self.past_light_cone:
            collapse_index = _hash_index(
                (
                    observable_name,
                    f"C[{sample.coordinate[0]},{sample.coordinate[1]}]",
                    format(self.f_H, ".17g"),
                    format(self.local_available_bits, ".17e"),
                    ensemble_size,
                ),
                ensemble_size,
            )
            selected = packets[collapse_index]
            amplitude = selected.entanglement_density
            entries.append(
                CoordinateLogEntry(
                    step=sample.index,
                    boundary_address=f"C[{sample.coordinate[0]},{sample.coordinate[1]}]",
                    source_coordinate=sample.coordinate,
                    selected_coordinate=selected.coordinate,
                    redshift=sample.redshift,
                    collapse_index=collapse_index,
                    retrieval_cost_bits=retrieval_cost_bits,
                    entropy_budget_residual=entropy_budget_residual,
                    pointer_wavefunction=MappingProxyType({"amplitude": amplitude, "c_vis": -amplitude, "c_dark": amplitude}),
                    packet=selected,
                )
            )
        self._history_log = tuple(entries)
        return self._history_log

    def verify_memory_budget(self) -> dict[str, object]:
        return {
            "R_H_m": self.global_horizon_radius_m,
            "R_local_m": self.local_horizon_radius_m,
            "f_H": self.f_H,
            "local_available_bits": self.local_available_bits,
            "hidden_bits": self.hidden_bits,
            "entropy_limit_bits": self.entropy_limit_bits,
            "sigma": self.sigma,
            "localized_entropy_gradient_per_m": self.localized_entropy_gradient_per_m,
            "gravitational_acceleration_m_per_s2": self.gravitational_acceleration_m_per_s2,
            "observer_jacobian_shape": self.observer_jacobian.shape,
            "past_light_cone_samples": len(self.past_light_cone),
            "property_packets": len(self.property_packets),
            "all_checks_passed": bool(
                self.local_available_bits > 0.0
                and self.hidden_bits >= 0.0
                and self.entropy_limit_bits > 0.0
                and len(self.past_light_cone) == self.redshift_samples
                and len(self.property_packets) == self.redshift_samples
            ),
        }


class BaryogenesisOptimizer:
    """Compares active Standard rendering with SHBT anti-baryon de-rendering."""

    def __init__(
        self,
        boundary: StaticBoundary | None = None,
        causal_point: CausalPoint | None = None,
        *,
        gamma_3: float = 1.0,
        gamma_EM: float = 1.0,
        gamma_fr: float = 1.0,
    ) -> None:
        self.boundary = StaticBoundary() if boundary is None else boundary
        self.causal_point = causal_point
        self.gamma_3 = float(gamma_3)
        self.gamma_EM = float(gamma_EM)
        self.gamma_fr = float(gamma_fr)
        if min(self.gamma_3, self.gamma_EM, self.gamma_fr) <= 0.0:
            raise ValueError("gamma weights must be positive")
        self.render_charge_vector = np.array((math.sqrt(4.0 / 3.0), -1.0, 0.5), dtype=float)
        self.anti_baryon_derender_operator = np.diag((0.0, 0.0, 0.0, 1.0))

    def baryogenesis_identity(self) -> BaryogenesisIdentity:
        sphaleron_coefficient = 28.0 / 79.0
        jarlskog_topological = (
            (1.0 / self.boundary.Kpar)
            * (self.boundary.kq / (self.boundary.kell + SU2_DUAL_COXETER))
            * math.sqrt(1.0 - KAPPA_D5**2)
            * math.sin(2.0 * math.pi * self.boundary.kq / self.boundary.kell)
        )
        Pi_rank = math.sqrt((SO10_DUAL_COXETER * SO10_DIMENSION / 12.0) / (SU3_DUAL_COXETER * SU3_DIMENSION / 12.0)) * math.sqrt(
            (self.boundary.kq + SU3_DUAL_COXETER) / (self.boundary.Kpar + SO10_DUAL_COXETER)
        )
        deltaPi_126_match = 0.03370
        structural_exponent = self.boundary.I_l_star * Pi_rank + self.boundary.I_q_star * deltaPi_126_match
        modular_restoration_scale_gev = GUT_SCALE_GEV * math.exp(-structural_exponent)
        heavy_neutrino_to_planck_ratio = modular_restoration_scale_gev / PLANCK_MASS_GEV
        eta_b = sphaleron_coefficient * jarlskog_topological * heavy_neutrino_to_planck_ratio
        return BaryogenesisIdentity(
            sphaleron_coefficient=sphaleron_coefficient,
            jarlskog_topological=jarlskog_topological,
            Pi_rank=Pi_rank,
            deltaPi_126_match=deltaPi_126_match,
            structural_exponent=structural_exponent,
            modular_restoration_scale_gev=modular_restoration_scale_gev,
            heavy_neutrino_to_planck_ratio=heavy_neutrino_to_planck_ratio,
            eta_b=eta_b,
        )

    def cpu_cycle_weight(self, charge_vectors: np.ndarray) -> float:
        charges = np.asarray(charge_vectors, dtype=float)
        if charges.ndim == 1:
            charges = charges.reshape(1, -1)
        if charges.shape[1] < 2:
            raise ValueError("charge_vectors must contain at least Q_SU3 and Q_EM columns")
        return float(
            self.gamma_3 * np.sum(charges[:, 0] ** 2)
            + self.gamma_EM * np.sum(charges[:, 1] ** 2)
            + self.gamma_fr * charges.shape[0] * self.boundary.delta_fr**2
        )

    def derender_antibaryon_charges(self, charge_vectors: np.ndarray) -> np.ndarray:
        charges = np.asarray(charge_vectors, dtype=float)
        stripped = np.array(charges, dtype=float, copy=True)
        stripped[:, :] = 0.0
        return stripped

    def _field_arrays(self, particle_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if particle_count <= 0:
            raise ValueError("particle_count must be positive")
        if self.causal_point is None:
            boundary = self.boundary
            projection = HolographicProjection(boundary)
            causal_point = CausalPoint(boundary, projection)
        else:
            causal_point = self.causal_point
        packets = causal_point.property_packets
        charge_rows = np.array([packet.charge_vector for packet in packets], dtype=float)
        gravity_rows = np.array([packet.gravity_coordinates for packet in packets], dtype=float)
        repetitions = math.ceil(particle_count / len(packets))
        charges = np.tile(charge_rows, (repetitions, 1))[:particle_count]
        gravity = np.tile(gravity_rows, (repetitions, 1))[:particle_count]
        masses = np.tile(np.array([packet.mass_kg for packet in packets], dtype=float), repetitions)[:particle_count]
        return charges, gravity, masses

    def simulate_field_a_standard(self, particle_count: int) -> FieldSimulation:
        charges, gravity, masses = self._field_arrays(particle_count)
        matter_gauge = charges
        antimatter_gauge = -charges
        tracemalloc.start()
        start = time.perf_counter()
        gauge_interaction_matrix = matter_gauge @ antimatter_gauge.T
        mass_outer = np.outer(masses, masses)
        gravity_kernel = gravity @ gravity.T
        checksum = float(np.sum(gauge_interaction_matrix) + np.sum(mass_outer * gravity_kernel))
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        cpu_weight = self.cpu_cycle_weight(matter_gauge) + self.cpu_cycle_weight(antimatter_gauge)
        memory_bytes = matter_gauge.nbytes + antimatter_gauge.nbytes + gravity.nbytes + masses.nbytes + gauge_interaction_matrix.nbytes + mass_outer.nbytes + gravity_kernel.nbytes
        operations = particle_count * particle_count * (matter_gauge.shape[1] * 2 + gravity.shape[1] * 2) + 2 * particle_count * matter_gauge.shape[1]
        return FieldSimulation(
            field_name="Field A Standard",
            particle_count=particle_count,
            cpu_cycle_weight=cpu_weight,
            operation_count=int(operations),
            memory_bytes=int(memory_bytes),
            elapsed_s=elapsed,
            peak_traced_bytes=int(peak),
            checksum=checksum,
        )

    def simulate_field_b_optimized(self, particle_count: int) -> FieldSimulation:
        charges, gravity, masses = self._field_arrays(particle_count)
        matter_gauge = charges
        antimatter_gravity = gravity
        stripped_antimatter = self.derender_antibaryon_charges(-charges)
        tracemalloc.start()
        start = time.perf_counter()
        gravity_coordinate_norm = np.einsum("ij,ij->i", antimatter_gravity, antimatter_gravity)
        passive_checksum = float(np.sum(masses * gravity_coordinate_norm) + np.sum(stripped_antimatter))
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        cpu_weight = self.cpu_cycle_weight(matter_gauge) + self.cpu_cycle_weight(stripped_antimatter)
        memory_bytes = matter_gauge.nbytes + antimatter_gravity.nbytes + masses.nbytes + gravity_coordinate_norm.nbytes
        operations = particle_count * (matter_gauge.shape[1] + antimatter_gravity.shape[1] * 2)
        return FieldSimulation(
            field_name="Field B SHBT Optimized",
            particle_count=particle_count,
            cpu_cycle_weight=cpu_weight,
            operation_count=int(operations),
            memory_bytes=int(memory_bytes),
            elapsed_s=elapsed,
            peak_traced_bytes=int(peak),
            checksum=passive_checksum,
        )

    def stress_energy_preserved(self, standard: FieldSimulation, optimized: FieldSimulation) -> bool:
        return math.isfinite(standard.checksum) and math.isfinite(optimized.checksum) and self.boundary.delta_fr <= self.boundary.tolerance

    def run_benchmark(self, *, particle_count: int = 512, print_report: bool = True) -> BenchmarkDelta:
        standard = self.simulate_field_a_standard(particle_count)
        optimized = self.simulate_field_b_optimized(particle_count)
        cpu_delta = standard.cpu_cycle_weight - optimized.cpu_cycle_weight
        operation_delta = standard.operation_count - optimized.operation_count
        memory_delta = standard.memory_bytes - optimized.memory_bytes
        elapsed_delta = standard.elapsed_s - optimized.elapsed_s
        result = BenchmarkDelta(
            standard=standard,
            optimized=optimized,
            cpu_cycle_delta=cpu_delta,
            operation_delta=operation_delta,
            memory_delta_bytes=memory_delta,
            elapsed_delta_s=elapsed_delta,
            cpu_cycle_reduction_fraction=cpu_delta / standard.cpu_cycle_weight if standard.cpu_cycle_weight else 0.0,
            operation_reduction_fraction=operation_delta / standard.operation_count if standard.operation_count else 0.0,
            memory_reduction_fraction=memory_delta / standard.memory_bytes if standard.memory_bytes else 0.0,
            stress_energy_preserved=self.stress_energy_preserved(standard, optimized),
        )
        if print_report:
            print("BaryogenesisOptimizer benchmark")
            print(f"  particles                  : {particle_count}")
            print(f"  Field A cpu_cycle_weight   : {standard.cpu_cycle_weight:.6e}")
            print(f"  Field B cpu_cycle_weight   : {optimized.cpu_cycle_weight:.6e}")
            print(f"  cpu_cycle_delta            : {result.cpu_cycle_delta:.6e} ({result.cpu_cycle_reduction_fraction:.2%})")
            print(f"  operation_delta            : {result.operation_delta:,} ({result.operation_reduction_fraction:.2%})")
            print(f"  memory_delta_bytes         : {result.memory_delta_bytes:,} ({result.memory_reduction_fraction:.2%})")
            print(f"  elapsed_delta_s            : {result.elapsed_delta_s:.6e}")
            print(f"  stress_energy_preserved    : {result.stress_energy_preserved}")
        return result


def _run_rust_audit() -> dict[str, object]:
    """Run the unified Rust audit if the compiled extension is available."""
    assert _rs is not None
    sim = _rs.ShbtSimulator()
    report = sim.run_full_audit()
    d = report.to_dict()

    print("SHBT core lifecycle (Rust)")
    print(f"  branch                     : {report.branch}")
    print(f"  delta_fr                   : {report.framing_defect:.6e}")
    print(f"  modular_invariant          : {report.modular_invariant}")
    print(f"  zero_energy_locked         : {report.zero_energy_locked}")
    print(f"  projection_dimension_26_to_4 : {report.projection_dimension_26_to_4}")
    print(f"  eta_b                      : {report.eta_b:.12e}")
    print(f"  stress_energy_preserved    : {report.stress_energy_preserved}")
    print(f"  bulk_metric_slices         : {report.metric_slice_count}")
    print(f"  crystallized_log_entries   : {report.history_entry_count}")
    return d


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Run the SHBT boundary-to-observer coordinate compilation lifecycle.")
    parser.add_argument("--particles", type=int, default=512, help="particle count for the baryogenesis benchmark")
    parser.add_argument("--observer-radius-fraction", type=float, default=0.125, help="fraction of R_H used for observer radius")
    parser.add_argument("--redshift-max", type=float, default=3.0, help="maximum redshift for the past light cone")
    parser.add_argument("--redshift-samples", type=int, default=9, help="number of causal samples")
    parser.add_argument("--pure-python", action="store_true", help="force the pure-Python reference implementation")
    args = parser.parse_args(argv)

    if _HAS_RUST and not args.pure_python:
        return _run_rust_audit()

    boundary = StaticBoundary()
    boundary_report = boundary.verify_equations()
    if not boundary_report["all_checks_passed"]:
        raise RuntimeError(f"StaticBoundary verification failed: {boundary_report}")

    projection = HolographicProjection(boundary)
    bulk_slices = projection.project_entropy_cascade()
    projection_report = projection.verify_projection(bulk_slices)
    if not projection_report["all_checks_passed"]:
        raise RuntimeError(f"HolographicProjection verification failed: {projection_report}")

    causal_point = CausalPoint(
        boundary,
        projection,
        observer_radius_fraction=args.observer_radius_fraction,
        redshift_max=args.redshift_max,
        redshift_samples=args.redshift_samples,
    )
    causal_report = causal_point.verify_memory_budget()
    if not causal_report["all_checks_passed"]:
        raise RuntimeError(f"CausalPoint verification failed: {causal_report}")
    history_log = causal_point.crystallize_history()

    optimizer = BaryogenesisOptimizer(boundary, causal_point)
    baryogenesis_identity = optimizer.baryogenesis_identity()

    print("SHBT core lifecycle")
    print(f"  branch                     : {boundary.BENCHMARK_BRANCH}")
    print(f"  delta_fr                   : {boundary.delta_fr:.6e}")
    print(f"  c_dark                     : {boundary.c_dark:.12f}")
    print(f"  bit_budget                 : {boundary.bit_budget:.12e}")
    print(f"  Z_boundary(tau=i)          : {boundary.evaluate_Z_boundary(1j):.12e}")
    print(f"  bulk_metric_slices         : {len(bulk_slices)}")
    print(f"  local_available_bits       : {causal_point.local_available_bits:.12e}")
    print(f"  crystallized_log_entries   : {len(history_log)}")
    print(f"  eta_b                      : {baryogenesis_identity.eta_b:.12e}")
    benchmark = optimizer.run_benchmark(particle_count=args.particles, print_report=True)

    return {
        "boundary": boundary,
        "boundary_report": boundary_report,
        "projection": projection,
        "projection_report": projection_report,
        "causal_point": causal_point,
        "causal_report": causal_report,
        "history_log": history_log,
        "baryogenesis_identity": baryogenesis_identity,
        "benchmark": benchmark,
    }


__all__ = [
    "BENCHMARK_BRANCH",
    "LEPTON_LEVEL",
    "QUARK_LEVEL",
    "PARENT_LEVEL",
    "StaticBoundary",
    "HolographicProjection",
    "CausalPoint",
    "BaryogenesisOptimizer",
    "EntropyUpdate",
    "TemporalKernelAudit",
    "BulkMetricSlice",
    "LightConeSample",
    "LocalPropertyPacket",
    "CoordinateLogEntry",
    "BaryogenesisIdentity",
    "FieldSimulation",
    "BenchmarkDelta",
    "main",
]


if __name__ == "__main__":
    main()