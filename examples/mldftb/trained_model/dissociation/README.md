# Water fragmentation evaluation

Both trained models were evaluated on every frame of the neutral (30 frames) and positive (42 frames) reference trajectories, at electronic smearing widths 0.025 and 0.0025 eV. Plots:

- [0.025 eV](dissociation_0.025_eV.png), [PDF](dissociation_0.025_eV.pdf)
- [0.0025 eV](dissociation_0.0025_eV.png), [PDF](dissociation_0.0025_eV.pdf)

Each figure shows neutral and positive fragmentation, with internal energy above and both fragment charges below. Each energy curve is shifted by its own last-frame energy. DFT curves use the supplied AIMS energies and atomic multipoles, without changing their electronic treatment.

## Inputs and conventions

Source: `/Users/will/PhDwork/scf/metal_water_interfaces/all_download_csd3/metal_water/data/test_cases`.

The source files lack N_alpha/N_beta and electronic smearing. Each model's effective nuclear charges (H=1, O=2 here) and the supplied total charge set the electron count, split equally between spins: N_alpha=N_beta=14 for neutral H14O7, and 16 for positive H17O8. Model temperature units are respected when setting the requested width.

Following the supplied `fragmentation.py`, membership is fixed by the sign of x in the final frame, and distance is between atoms 4 and 6 (neutral) or 0 and 24 (positive), with zero-based indexing. Left fragments are H4O2 in both cases. Right fragments are H10O5 (neutral) and H13O6 (positive). The distances are the reference script's atom-pair distances, not centre-of-mass distances. Full atom memberships and model paths are recorded in `metadata.json`.

Predicted charge is the sum of the model's signed atomic monopoles, including effective nuclei. Both fragment charges sum to the specified configuration charge; all evaluations passed a 1e-7 tolerance check.

## Results at maximum separation

| Case / model | Left charge at 0.025 eV | Right charge at 0.025 eV |
|---|---:|---:|
| Neutral / multipole fit | -0.004066 | +0.004066 |
| Neutral / energy-force only | -0.000293 | +0.000293 |
| Positive / multipole fit | +0.009912 | +0.990088 |
| Positive / energy-force only | +0.003560 | +0.996440 |

At 0.0025 eV, both models give neutral charges 0/0 and positive charges 0/+1 to numerical precision. The reference endpoint charges differ from those integers by only a few millionths of an electron.

These particular water-fragment tests therefore do not show persistent large fractional charge at separation. The small endpoint deviations at 0.025 eV disappear at lower smearing. This does not establish robustness against another chemical species with substantially different orbital alignment.

Energy/force-only gives the closer energy curves overall at 0.025 eV: relative-energy RMSE is 0.088 eV (neutral) and 0.091 eV (positive), versus 0.142 and 0.196 eV for the multipole fit. These RMSE values include the most compressed geometry, where errors are largest. The multipole fit better captures much of the finite-distance charge redistribution; energy/force-only transfers much less charge than DFT in the neutral scan and localizes the positive charge earlier. Accurate energy curves alone therefore do not ensure accurate intermediate fragment charges.

## Files and reproduction

`results.csv` contains every model/width/frame energy, reference energy, both fragment charges, gap and alpha chemical potential. The eight evaluated XYZ files preserve source labels and add `MLDFTB_energy`, `MLDFTB_density_coefficients`, and the electronic input labels used for evaluation. Unlike orbital exports, these coefficients are the full configuration's multipoles.

From the repository root:

```bash
PYTHONPATH=. python scripts/evaluate_mldftb_dissociation.py \
  --data /Users/will/PhDwork/scf/metal_water_interfaces/all_download_csd3/metal_water/data/test_cases \
  --models examples/mldftb/trained_model/positive/mldftb_rewrite_initial.model examples/mldftb/trained_model/models/energy_force_only.model \
  --labels multipole_fit energy_force_only \
  --output examples/mldftb/trained_model/dissociation
```

The default widths are 0.025 and 0.0025 eV. The script uses CPU float64 with one PyTorch thread by default. Original datasets and models are unchanged.
