# Figure Captions

**Figure 1. End-to-end pipeline.** CHB-MIT EDF recordings are processed into per-EDF shards, loaded lazily for patient-wise training, evaluated with frozen metrics, and summarized into paper artifacts. The diagram documents the audited workflow and does not imply clinical deployment.

**Figure 2. Data flow.** Data move from raw EDF files to interim shard files, patient-wise metadata splits, model checkpoints, metric files, and publication tables. The figure emphasizes that the completed training used sharded data rather than a monolithic training array.

**Figure 3. Preprocessing flowchart.** Frozen preprocessing steps: required-channel check, channel ordering, 0.5--40 Hz bandpass filtering, 50 Hz notch filtering, 4 s windows with 2 s stride, per-window z-score normalization, and per-EDF shard writing.

**Figure 4. Training pipeline.** The training workflow combines patient-wise split metadata, lazy shard loading, model selection, AdamW optimization, mixed precision, early stopping, checkpointing, threshold selection, and final evaluation.

**Figure 5. Patient-wise split.** Train, validation, and test partitions contain disjoint patients. This split prevents patient identity leakage, although it does not replace repeated patient-wise or leave-one-patient-out validation.

**Figure 6. Dataset statistics by patient.** Window and label counts vary across patients, showing both seizure-frequency and recording-duration imbalance. These differences motivate patient-wise reporting and conservative interpretation.

**Figure 7. Positive and negative distribution.** The baseline 10 min preictal definition yields 14,145 positive and 720,651 negative windows, corresponding to 1.9250% positive prevalence.

**Figure 8. Distance to seizure onset.** Positive windows are distributed across the 10 min horizon; 47.26% occur more than 5 min before seizure onset. This supports concern that the positive class may mix early and late preictal physiology.

**Figure 9. Model comparison.** Validation PR-AUC and F1 for the final model runs. Values remain close to positive prevalence; the Hybrid run is shown for transparency but was incomplete.

**Figure 10. EEGNet ROC curve.** Window-level ROC curve for the final EEGNet evaluation. ROC-AUC is reported as a secondary metric because ROC curves can appear less sensitive to rare positive prevalence.

**Figure 11. EEGNet precision-recall curve.** Window-level precision-recall curve for the final EEGNet evaluation. PR-AUC is the primary ranking metric because positives are rare.

**Figure 12. EEGNet confusion matrix.** Threshold-dependent test-set confusion matrix for the selected EEGNet operating point. It should be interpreted together with PR-AUC and recall because threshold choice strongly affects counts.

**Figure 13. Learning curves.** EEGNet training history from the completed run. The curves document optimization behavior but do not by themselves establish generalization beyond the frozen split.

**Figure 14. Validation curves.** EEGNet validation metrics over training. These curves are used to inspect early stopping and threshold-dependent behavior.

**Figure 15. Preictal horizon PR-AUC.** Validation PR-AUC for 10, 7.5, 5, and 3 min horizons. Shorter global horizons did not improve PR-AUC over the 10 min baseline.

**Figure 16. Preictal horizon F1.** Validation F1 for 10, 7.5, 5, and 3 min horizons. Shorter global horizons did not improve F1 over the 10 min baseline.

**Figure 17. Positive class size versus horizon.** Shortening the preictal horizon monotonically reduces the number of positive windows, creating a trade-off between possible label specificity and positive sample count.
