# Final Submission Checklist

## Scientific Claims

- [x] No state-of-the-art claim.
- [x] No clinical-readiness claim.
- [x] Negative results framed as bounded by the frozen protocol.
- [x] "Validated" language replaced with "internally audited" or equivalent.
- [x] Label audit described as no evidence of mechanical contamination under implemented checks.
- [x] Horizon ablation described as not supporting the shorter-horizon hypothesis under this setup.

## Evidence Traceability

- [x] Dataset counts trace to `paper/tables/dataset_summary.csv`.
- [x] Patient split traces to `paper/tables/patient_split_table.csv`.
- [x] Final model metrics trace to `paper/tables/model_comparison_table.csv` and overnight metric JSON files.
- [x] Label audit findings trace to `outputs/paper/label_audit/label_audit_summary.json`.
- [x] Horizon ablation traces to `paper/tables/preictal_horizon_ablation_table.csv`.
- [x] No fabricated confidence intervals or repeated-seed statistics were added.

## Presentation

- [x] Abstract rewritten with Background, Objective, Methods, Results, and Conclusion structure.
- [x] Results separate observations from interpretations.
- [x] Limitations are explicit and separate from future work.
- [x] Figure captions improved.
- [x] Table captions improved.
- [x] References expanded with seizure prediction, EEGNet, imbalance, and PR-AUC sources.

## Remaining Submission Risks

- [ ] No LOPO or repeated patient-wise split.
- [ ] No confidence intervals.
- [ ] No seizure-level false-alarm metrics.
- [ ] Hybrid run incomplete.
- [ ] 10-patient subset limits generalizability.
- [ ] No classical feature-based baseline.
