```
"""Overlapping atom-centred principal-submatrix solves of a saved Hamiltonian.

The shared chemical potential is fitted to the sum of central-atom populations.
Each atom contributes only its own core block and band-energy row, avoiding
multiple counting of overlapping buffers. No model evaluation is performed.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import expit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Locality output directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--radii", nargs="+", type=float, default=[3, 4, 5, 6, 6.8, 8, 10]
    )
    parser.add_argument("--skip-full-radius", action="store_true")
    parser.add_argument(
        "--repeat-222-cores",
        type=int,
        help="Use this many primitive-cell atoms as representatives of an exact 2x2x2 replication; verify Hamiltonian translation symmetry",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    saved = np.load(args.input / "density_matrices.npz")
    metadata = json.loads((args.input / "summary.json").read_text())
    h = saved["hamiltonian"]
    distances = saved["distances"]
    n, b = metadata["atoms"], metadata["block_size"]
    from ase.io import read

    atoms = read(metadata["xyz"])
    counts = np.array([atoms.info["N_alpha"], atoms.info["N_beta"]], dtype=float)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.input / "hamiltonian.npy", h)
    results = []
    radii = sorted(
        set(
            args.radii
            + ([] if args.skip_full_radius else [float(distances.max()) + 1e-6])
        )
    )
    cores = args.repeat_222_cores or n
    multiplicity = n // cores
    symmetry_error = 0.0
    if args.repeat_222_cores:
        if n != 8 * cores:
            raise ValueError(
                "Representative mode requires exactly eight identical cells"
            )
        h4 = h.reshape(8, cores * b, 8, cores * b)
        for tile in range(8):
            symmetry_error = max(
                symmetry_error,
                float(np.max(np.abs(h4[tile][:, np.arange(8) ^ tile, :] - h4[0]))),
            )
            if symmetry_error > 1e-8:
                raise ValueError(
                    "Hamiltonian is not translation-equivalent under 2x2x2 replication"
                )
    references = {}
    for ref in metadata["results"]:
        pref = saved[f"density_{ref['width_eV']:g}"]
        gamma_ref = np.stack(
            [pref[i * b : (i + 1) * b, i * b : (i + 1) * b] for i in range(n)]
        )
        if (
            multiplicity > 1
            and np.max(
                np.abs(gamma_ref.reshape(multiplicity, cores, b, b) - gamma_ref[:cores])
            )
            > 1e-8
        ):
            raise ValueError("Reference onsite density violates translation symmetry")
        references[ref["width_eV"]] = (
            gamma_ref[:cores],
            float(np.einsum("ij,ji->", pref, h)),
        )
        del pref
    htorch = torch.from_numpy(h)
    for radius in radii:
        start = time.perf_counter()
        spectra, central_vectors, weights, sizes = [], [], [], []
        # Cache identical atom sets (all full-system fragments share one solve).
        eigencache = {}
        for i in range(cores):
            neighbours = np.flatnonzero(distances[i] <= radius)
            key = tuple(neighbours)
            indices = (neighbours[:, None] * b + np.arange(b)).ravel()
            core = int(np.flatnonzero(neighbours == i)[0]) * b
            if key in eigencache:
                e, u = eigencache[key]
            else:
                index = torch.from_numpy(indices)
                e, u = torch.linalg.eigh(htorch[index[:, None], index[None, :]])
                if len(neighbours) == n:
                    eigencache[key] = (e, u)
            uc = u[core : core + b].numpy().copy()
            spectra.append(e.numpy().copy())
            central_vectors.append(uc)
            weights.append(np.sum(uc**2, axis=0))
            sizes.append(len(indices))
        solve_seconds = time.perf_counter() - start
        # Pad for fast shared-mu evaluation; fictitious states have zero core weight.
        energies = np.zeros((cores, max(sizes)))
        coreweights = np.zeros_like(energies)
        for i, (e, w) in enumerate(zip(spectra, weights)):
            energies[i, : len(e)] = e
            coreweights[i, : len(e)] = w
        for reference in metadata["results"]:
            tau = reference["width_eV"]
            gamma_ref, band_ref = references[tau]
            for mode in ["reference_mu", "fitted_mu"]:
                fill_start = time.perf_counter()
                mus = []
                for spin, count in enumerate(counts):
                    if mode == "reference_mu":
                        mu = reference["chemical_potentials"][spin]
                    else:
                        lower = min(float(e.min()) for e in spectra) - 100 * tau - 1
                        upper = max(float(e.max()) for e in spectra) + 100 * tau + 1
                        mu = brentq(
                            lambda mu: np.sum(
                                coreweights * expit((mu - energies) / tau)
                            )
                            - count / multiplicity,
                            lower,
                            upper,
                            xtol=1e-14,
                        )
                    mus.append(mu)
                filling = expit((mus[0] - energies) / tau) + expit(
                    (mus[1] - energies) / tau
                )
                gamma = np.stack(
                    [
                        (uc * filling[i, : sizes[i]]) @ uc.T
                        for i, uc in enumerate(central_vectors)
                    ]
                )
                # H_f U=U eps -> sum_{a in core,b in fragment} P_ab H_ba
                band = multiplicity * float(np.sum(coreweights * filling * energies))
                error = gamma - gamma_ref
                row = dict(
                    radius_A=radius,
                    representative_cores=cores,
                    translational_multiplicity=multiplicity,
                    width_eV=tau,
                    mu_mode=mode,
                    mean_fragment_atoms=float(np.mean(sizes) / b),
                    max_fragment_atoms=max(sizes) // b,
                    fragment_solves_seconds=solve_seconds,
                    filling_seconds=time.perf_counter() - fill_start,
                    chemical_potential_alpha=mus[0],
                    chemical_potential_beta=mus[1],
                    electron_count_error=float(
                        multiplicity * np.sum(coreweights * filling) - counts.sum()
                    ),
                    max_onsite_entry_error=float(np.abs(error).max()),
                    rms_onsite_entry_error=float(np.sqrt(np.mean(error**2))),
                    max_atom_population_error=float(
                        np.abs(np.trace(error, axis1=1, axis2=2)).max()
                    ),
                    band_energy_error_eV=band - band_ref,
                    band_energy_error_meV_per_atom=1000 * (band - band_ref) / n,
                )
                results.append(row)
                with (args.output / "results.jsonl").open(
                    "a" if len(results) > 1 else "w"
                ) as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        if radius >= distances.max():
            for row in results[-2 * len(metadata["results"]) :]:
                if (
                    row["max_onsite_entry_error"] > 1e-8
                    or abs(row["band_energy_error_eV"]) > 1e-7
                ):
                    raise RuntimeError("Full-fragment reference check failed")
    with (args.output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (args.output / "settings.json").write_text(
        json.dumps(
            dict(
                input=str(args.input),
                radii=radii,
                counts=counts.tolist(),
                representative_cores=cores,
                hamiltonian_translation_error_eV=symmetry_error,
                threads=1,
                model=metadata["model"],
            ),
            indent=2,
        )
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fields = [
        "max_onsite_entry_error",
        "max_atom_population_error",
        "band_energy_error_meV_per_atom",
        "electron_count_error",
    ]
    labels = [
        "Maximum onsite density entry error",
        "Maximum atom population error (electrons)",
        "Band energy error (meV/atom)",
        "Total electron-count error",
    ]
    for ref in metadata["results"]:
        for mode in ["reference_mu", "fitted_mu"]:
            rows = [
                r
                for r in results
                if r["width_eV"] == ref["width_eV"] and r["mu_mode"] == mode
            ]
            for ax, field in zip(axes.flat, fields):
                ax.plot(
                    [r["radius_A"] for r in rows],
                    [r[field] for r in rows],
                    "-o" if mode == "fitted_mu" else "--",
                    label=f"{ref['width_eV']:g} eV, {mode}",
                )
    for ax, label in zip(axes.flat, labels):
        ax.set(xlabel="Fragment radius (Å)", ylabel=label, yscale="symlog")
        ax.set_yscale("symlog", linthresh=1e-7)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Overlapping fragments: only central-atom contributions retained")
    fig.tight_layout()
    fig.savefig(args.output / "fragment_errors.png", dpi=180)


if __name__ == "__main__":
    main()

```