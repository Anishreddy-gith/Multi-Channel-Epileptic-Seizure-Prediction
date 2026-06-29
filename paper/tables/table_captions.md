# Table Captions

**Table 1. Dataset summary.** Counts for the frozen CHB-MIT subset under the baseline 10 min preictal definition. The table reports windows, labels, channel count, and positive prevalence.

**Table 2. Patient-wise split.** Train, validation, and test partitions with disjoint patient IDs. The table supports the no-patient-leakage claim but does not replace repeated patient-wise validation.

**Table 3. Class distribution by patient.** Per-patient positive and negative window counts. The table shows substantial patient-level variation in positive prevalence.

**Table 4. Model configuration.** Architecture family, input shape, parameter count, and completion status for EEGNet, Attention, and Hybrid.

**Table 5. Training configuration.** Shared optimization settings used in the completed experiments, including seed, batch size, learning rate, weight decay, optimizer, scheduler, and AMP status.

**Table 6. Final model comparison.** Validation and test metrics for final model runs. Hybrid is included for transparency and marked incomplete.

**Table 7. Optimization summary.** Before/after optimization-cycle results. The after-run was not promoted because it did not improve both validation PR-AUC and validation F1.

**Table 8. Label audit summary.** Mechanical label-contamination checks and redundancy statistics from the label audit.

**Table 9. Preictal horizon dataset statistics.** Positive and negative window counts for each tested horizon. Shorter horizons reduce positives and prevalence.

**Table 10. Preictal horizon ablation.** EEGNet validation and test metrics for 10, 7.5, 5, and 3 min horizons. Results support only the conclusion that shorter global horizons did not improve both validation PR-AUC and validation F1 under this protocol.

**Table 11. Hardware and software configuration.** Machine, GPU, Python, CUDA, and package versions used for the completed experiments.
