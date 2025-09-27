#!/usr/bin/env python3
"""Evaluate Ta ADP energy/force/stress using tabulated data from Ta.adp.txt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from ase import Atoms, units
from ase.calculators.eam import EAM

from torch_neigh import TorchNeighborList


@dataclass
class UniformFunction:
    values: np.ndarray
    spacing: float
    zero_outside: bool

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        self.grid = np.arange(len(self.values), dtype=float) * self.spacing
        # Use second-order accurate finite differences for smooth derivatives
        self.derivatives = np.gradient(self.values, self.spacing, edge_order=2)

    def _clip(self, x: np.ndarray | float) -> np.ndarray:
        return np.clip(x, self.grid[0], self.grid[-1])

    def evaluate(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        clipped = self._clip(x)
        out = np.interp(clipped, self.grid, self.values)
        if self.zero_outside:
            out = np.where(x <= self.grid[-1], out, 0.0)
        return out

    def derivative(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        clipped = self._clip(x)
        out = np.interp(clipped, self.grid, self.derivatives)
        if self.zero_outside:
            out = np.where(x <= self.grid[-1], out, 0.0)
        return out


@dataclass
class ADPPotential:
    drho: float
    nrho: int
    F: UniformFunction
    dr: float
    nr: int
    rho: UniformFunction
    phi: UniformFunction
    u: UniformFunction
    w: UniformFunction
    rc: float
    mass: float


def _read_values(iterator: Iterable[str], count: int) -> np.ndarray:
    values: list[float] = []
    while len(values) < count:
        line = next(iterator)
        stripped = line.strip()
        if not stripped:
            continue
        values.extend(map(float, stripped.split()))
    return np.array(values[:count], dtype=float)


def load_adp_setfl(path: Path) -> ADPPotential:
    with path.open("r", encoding="utf-8") as f:
        lines = iter(f)
        # Skip three comment lines
        for _ in range(3):
            next(lines)

        parts = next(lines).split()
        nelements = int(parts[0])
        if nelements != 1:
            raise ValueError("This script currently supports single-element ADP files only.")

        element = parts[1]
        header = next(lines).split()
        nrho, drho = int(header[0]), float(header[1])
        nr, dr, rc = int(header[2]), float(header[3]), float(header[4])

        # Element block
        info = next(lines).split()
        _, mass = int(info[0]), float(info[1])

        F_rho = _read_values(lines, nrho)
        rho_r = _read_values(lines, nr)

        # Pair blocks (only Ta-Ta)
        phi_r = _read_values(lines, nr)
        u_r = _read_values(lines, nr)
        w_r = _read_values(lines, nr)

    r_grid = np.arange(nr, dtype=float) * dr
    phi_values = np.zeros_like(phi_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_values[1:] = phi_r[1:] / r_grid[1:]
    phi_values[0] = phi_values[1]

    return ADPPotential(
        drho=drho,
        nrho=nrho,
        F=UniformFunction(F_rho, drho, zero_outside=False),
        dr=dr,
        nr=nr,
        rho=UniformFunction(rho_r, dr, zero_outside=True),
        phi=UniformFunction(phi_values, dr, zero_outside=True),
        u=UniformFunction(u_r, dr, zero_outside=True),
        w=UniformFunction(w_r, dr, zero_outside=True),
        rc=rc,
        mass=mass,
    )


def read_ovito_lammps(path: Path) -> Atoms:
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    idx = 0

    def next_data_line() -> str:
        nonlocal idx
        while idx < len(lines):
            stripped = lines[idx].strip()
            idx += 1
            if stripped:
                return stripped
        raise ValueError("Unexpected end of file while reading LAMMPS data")

    line = next_data_line()
    while line.startswith("#"):
        line = next_data_line()

    natoms = int(line.split()[0])
    next_data_line()  # skip "atom types" line

    def parse_vector_line(expected_keyword: str) -> np.ndarray:
        vec_line = next_data_line()
        while expected_keyword not in vec_line:
            vec_line = next_data_line()
        parts = vec_line.split()
        return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)

    avec = parse_vector_line("avec")
    bvec = parse_vector_line("bvec")
    cvec = parse_vector_line("cvec")
    origin = parse_vector_line("origin")

    # Skip until Atoms section
    section_line = next_data_line()
    while not section_line.lower().startswith("atoms"):
        section_line = next_data_line()

    positions: list[list[float]] = []
    while len(positions) < natoms:
        atom_line = next_data_line()
        if atom_line.lower().startswith("velocities"):
            break
        if atom_line.startswith("#"):
            continue
        tokens = atom_line.split()
        if len(tokens) < 5:
            continue
        positions.append([float(tokens[2]), float(tokens[3]), float(tokens[4])])

    positions_array = np.array(positions, dtype=float)
    if len(positions_array) != natoms:
        raise ValueError(
            f"Expected {natoms} atoms, found {len(positions_array)} atoms in data file"
        )

    positions_array += origin
    cell = np.vstack([avec, bvec, cvec])
    symbols = ["Ta"] * natoms
    return Atoms(symbols=symbols, positions=positions_array, cell=cell, pbc=True)


def compute_adp(atoms_path: Path, adp: ADPPotential) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    atoms = read_ovito_lammps(atoms_path)

    cutoff = adp.rc + 1e-6
    neighbor_list = TorchNeighborList(cutoff=cutoff)
    pairs, pair_diff, pair_dist = neighbor_list(atoms)

    mask = pairs[:, 0] < pairs[:, 1]
    pair_i = pairs[mask, 0].astype(int)
    pair_j = pairs[mask, 1].astype(int)
    pair_vec = pair_diff[mask]
    pair_dist = pair_dist[mask]

    n_atoms = len(atoms)
    density = np.zeros(n_atoms)
    mu = np.zeros((n_atoms, 3))
    lam = np.zeros((n_atoms, 3, 3))

    rho_vals = adp.rho.evaluate(pair_dist)
    phi_vals = adp.phi.evaluate(pair_dist)
    u_vals = adp.u.evaluate(pair_dist)
    w_vals = adp.w.evaluate(pair_dist)

    pair_energy = 0.0
    for idx in range(len(pair_i)):
        i = pair_i[idx]
        j = pair_j[idx]
        rij = pair_dist[idx]
        if rij <= 1e-10:
            continue
        del_vec = -pair_vec[idx]  # x_i - x_j

        rho = rho_vals[idx]
        density[i] += rho
        density[j] += rho

        pair_energy += phi_vals[idx]

        u_val = u_vals[idx]
        mu[i] += u_val * del_vec
        mu[j] -= u_val * del_vec

        w_val = w_vals[idx]
        outer = w_val * np.outer(del_vec, del_vec)
        lam[i] += outer
        lam[j] += outer

    embed_vals = adp.F.evaluate(density)
    embedding_energy = float(np.sum(embed_vals))
    Fp = adp.F.derivative(density)

    mu_sq = np.sum(mu ** 2, axis=1)
    lam_diag = np.stack((lam[:, 0, 0], lam[:, 1, 1], lam[:, 2, 2]), axis=1)
    lam_diag_sq = np.sum(lam_diag ** 2, axis=1)
    lam_off_sq = lam[:, 0, 1] ** 2 + lam[:, 0, 2] ** 2 + lam[:, 1, 2] ** 2
    nu = np.sum(lam_diag, axis=1)
    adp_per_atom = 0.5 * mu_sq + 0.5 * lam_diag_sq + lam_off_sq - (nu ** 2) / 6.0
    adp_energy = float(np.sum(adp_per_atom))

    forces = np.zeros((n_atoms, 3))
    virial = np.zeros((3, 3))

    rho_derivs = adp.rho.derivative(pair_dist)
    phi_derivs = adp.phi.derivative(pair_dist)
    u_derivs = adp.u.derivative(pair_dist)
    w_derivs = adp.w.derivative(pair_dist)

    for idx in range(len(pair_i)):
        i = pair_i[idx]
        j = pair_j[idx]
        rij = pair_dist[idx]
        if rij <= 1e-10:
            continue
        del_vec = -pair_vec[idx]
        rinv = 1.0 / rij

        rho_prime = rho_derivs[idx]
        phi_prime = phi_derivs[idx]
        u_val = u_vals[idx]
        u_prime = u_derivs[idx]
        w_val = w_vals[idx]
        w_prime = w_derivs[idx]

        fpair = -(phi_prime + Fp[i] * rho_prime + Fp[j] * rho_prime)
        radial = fpair * rinv * del_vec

        delmux = mu[i] - mu[j]
        trdelmu = float(np.dot(delmux, del_vec))
        lam_sum = lam[i] + lam[j]

        sumlamxx = lam_sum[0, 0]
        sumlamyy = lam_sum[1, 1]
        sumlamzz = lam_sum[2, 2]
        sumlamxy = lam_sum[0, 1]
        sumlamxz = lam_sum[0, 2]
        sumlamyz = lam_sum[1, 2]

        delx, dely, delz = del_vec
        tradellam = (
            sumlamxx * delx * delx
            + sumlamyy * dely * dely
            + sumlamzz * delz * delz
            + 2.0 * sumlamxy * delx * dely
            + 2.0 * sumlamxz * delx * delz
            + 2.0 * sumlamyz * dely * delz
        )
        nu_sum = sumlamxx + sumlamyy + sumlamzz

        adpx = (
            delmux[0] * u_val
            + trdelmu * u_prime * delx * rinv
            + 2.0 * w_val * (sumlamxx * delx + sumlamxy * dely + sumlamxz * delz)
            + w_prime * delx * rinv * tradellam
            - (1.0 / 3.0) * nu_sum * (w_prime * rij + 2.0 * w_val) * delx
        )
        adpy = (
            delmux[1] * u_val
            + trdelmu * u_prime * dely * rinv
            + 2.0 * w_val * (sumlamxy * delx + sumlamyy * dely + sumlamyz * delz)
            + w_prime * dely * rinv * tradellam
            - (1.0 / 3.0) * nu_sum * (w_prime * rij + 2.0 * w_val) * dely
        )
        adpz = (
            delmux[2] * u_val
            + trdelmu * u_prime * delz * rinv
            + 2.0 * w_val * (sumlamxz * delx + sumlamyz * dely + sumlamzz * delz)
            + w_prime * delz * rinv * tradellam
            - (1.0 / 3.0) * nu_sum * (w_prime * rij + 2.0 * w_val) * delz
        )

        adp_vec = -np.array([adpx, adpy, adpz])
        force_vec = radial + adp_vec

        forces[i] += force_vec
        forces[j] -= force_vec
        virial += np.outer(del_vec, force_vec)

    volume = atoms.get_volume()
    stress_tensor = virial / volume
    stress_voigt = np.array(
        [
            stress_tensor[0, 0],
            stress_tensor[1, 1],
            stress_tensor[2, 2],
            stress_tensor[1, 2],
            stress_tensor[0, 2],
            stress_tensor[0, 1],
        ]
    )

    total_energy = pair_energy + embedding_energy + adp_energy
    return total_energy, pair_energy, embedding_energy, adp_energy, forces, stress_voigt


def compute_with_ase(atoms_path: Path, potential_path: Path) -> Tuple[float, np.ndarray, np.ndarray]:
    atoms = read_ovito_lammps(atoms_path)
    calc = EAM(potential=str(potential_path), elements=["Ta"], form="adp")
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    try:
        stress = atoms.get_stress()  # ASE returns Voigt [xx, yy, zz, yz, xz, xy]
    except Exception:
        stress = None
    return energy, forces, stress


def main() -> None:
    root = Path(__file__).resolve().parent
    adp_path = root / "Ta.adp.txt"
    structure_path = root / "model_Ta_dump_idx10.dat"

    adp = load_adp_setfl(adp_path)
    total_energy, pair_energy, embed_energy, adp_energy, forces, stress_voigt = compute_adp(
        structure_path,
        adp,
    )

    print("=== Ta ADP results (analytic implementation) ===")
    print(f"Total energy:      {total_energy:.12f} eV")
    print(f"  Pair energy:     {pair_energy:.12f} eV")
    print(f"  Embedding energy:{embed_energy:.12f} eV")
    print(f"  ADP energy:      {adp_energy:.12f} eV")
    print()
    print("Forces (eV/Å):")
    print(forces)
    print()
    print("Stress (eV/Å^3) [xx, yy, zz, yz, xz, xy]:")
    print(stress_voigt)
    ev_per_a3_to_gpa = units.eV / (units.Angstrom ** 3 * units.GPa)
    print("Stress (GPa):")
    print(stress_voigt * ev_per_a3_to_gpa)

    print()
    print("=== Ta ADP results (ASE EAM calculator) ===")
    ase_energy, ase_forces, ase_stress = compute_with_ase(structure_path, adp_path)
    print(f"Total energy:      {ase_energy:.12f} eV")
    if ase_forces is not None:
        print("Forces (eV/Å):")
        print(ase_forces)
    if ase_stress is not None:
        print("Stress (eV/Å^3) [xx, yy, zz, yz, xz, xy]:")
        print(ase_stress)
        print("Stress (GPa):")
        print(ase_stress * ev_per_a3_to_gpa)


if __name__ == "__main__":
    main()
