# Training MLDFTB and using ASE

Edit `config.yaml` for your species and property names, then run from any directory:

```bash
bash /path/to/mace_scf/examples/mldftb/train.sh /path/to/train.xyz
# Optional overrides:
bash train.sh /path/to/train.xyz --device=cuda --valid_file=/path/to/valid.xyz
```

`PYTHON=/path/to/python` selects the environment. The script writes logs, checkpoints and models under `examples/mldftb/run` by default; override with `--work_dir`. No training dataset is included. Reusing a run with `--restart_latest` resumes its checkpoint; otherwise use a fresh work directory for independent experiments.

Every frame must provide `N_alpha` and `N_beta`. With the example's `elec_temp_units: eV`, set `elec_temp` to a finite smearing width in eV (e.g. `0.05`). Missing temperature defaults to zero, which is unsuitable at level crossings. Counts must satisfy `N_alpha + N_beta = sum(Z_eff) - total_charge`, with each spin count at most `number_of_atoms * (n_s + 3*n_p)`. Missing total charge defaults to zero. Example H/O charges are choices to edit, not inferred physical requirements.

Map energy/force labels to `REF_energy`/`REF_forces` or edit the YAML. No Fermi-level labels, SCF settings or reference multipoles are required. For periodic data, provide the cell and PBC flags; `auto` also supports cell-free molecules and mixed batches. To fit stress, enable `compute_stress`, its key mapping and its loss.

To supervise multipoles, uncomment their array mapping and `atomic_multipoles` loss. The existing loss uses signed `(charge, dipole_y, dipole_z, dipole_x)` targets in e and eÅ, including the nuclear monopole. It is unchanged. Missing labels have zero density-loss weight and are excluded from multipole metrics. For dipole supervision, also uncomment its mapping/loss and supply the existing required `config_dipole_weight` three-vector in each labelled frame. Optional per-atom nuclear charges require `nuclear_charge_mode: from_data` and the corresponding array mapping.

`mldftb_config` accepts the electronic/density options documented in [implementation_reference.md](../../ml_dftb/implementation_reference.md). Here `effective_nuclear_charges` is a dictionary keyed by **atomic number**; the factory orders it for the dataset's species table. Shared MACE options stay at the YAML top level. `sigma`, `pbc_handling` and `kspace_cutoff_factor` inside `mldftb_config` override the shared `atomic_multipoles_smearing_width`, `electrostatic_pbc_method` and `kspace_cutoff_factor` respectively. The optimizer includes every Hamiltonian parameter. Start with small batches: the eigensolver uses a dense matrix for each structure.

Models are saved under `run/checkpoints/<name>_run-<seed>_<stage>.model`; the training log prints the exact path. Use it with ASE:

```python
from ase.io import read
from mace_scf.calculators.mldftb import MLDFTBCalculator

atoms = read("structure.xyz")
# Supply counts/width here if not already present in the file:
# atoms.info.update(N_alpha=4.0, N_beta=4.0, elec_temp=0.05, total_charge=0.0)
atoms.calc = MLDFTBCalculator(
    model_path="path/from/training/log.model", device="cpu",
    # For nonstandard input names: N_alpha_key="na", N_beta_key="nb", ...
)
energy = atoms.get_potential_energy(force_consistent=True)
forces = atoms.get_forces()
charges = atoms.get_charges()
dipole = atoms.get_dipole_moment()
# stress = atoms.get_stress()  # ASE Voigt order, eV/Angstrom^3
```

The calculator expects eV/Å units and the saved model's temperature convention. It does not infer electron counts. Changes to electronic metadata or per-atom nuclear charges invalidate its cache, as do normal geometry changes. You may instead pass `model=model` for an in-memory model; the calculator takes a copy.

ASE `energy` is internal energy. ASE `free_energy` is the **selected training energy**, whose derivative gives the returned forces: internal for `energy_kind: internal`, free for `energy_kind: free`. Thus `force_consistent=True` selects the matching energy in both modes. `electronic_free_energy` in `calc.results` always includes the entropy term; named energy components and spin chemical potentials are also available there. Charged-system dipoles depend on origin, and periodic dipoles use the supplied coordinate representatives.
