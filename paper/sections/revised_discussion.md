# Revised Discussion

The central finding is a bounded negative result: under a patient-wise split and the completed preprocessing protocol, standard imbalance handling, threshold optimization, and the evaluated neural architectures did not produce reliable window-level seizure prediction.

The preprocessing and data audits matter because EEG seizure prediction pipelines are vulnerable to silent failure modes. If seizure windows, postictal windows, duplicate windows, or patient identities leak across splits, performance can appear artificially strong. The audit trail reduces this risk and makes the negative result more interpretable.

The label audit matters because poor performance across multiple model and loss configurations suggested that the bottleneck might be in the target definition. The audit found no evidence of mechanical label contamination, but it did show that nearly half of positive windows were more than five minutes before onset. This supports caution about broad fixed preictal labels, although the subsequent horizon ablation did not show that shortening the horizon alone improves performance.

The imbalance experiments matter because class prevalence is only 1.9250%. Accuracy is therefore uninformative, and even threshold-dependent metrics can be dominated by operating-point choices. PR-AUC and MCC indicate that the models remained close to prevalence-level ranking performance.

The negative findings are scientifically useful because they constrain future work. They suggest that simply adding capacity, reweighting losses, or shortening a global horizon may be insufficient for patient-wise CHB-MIT seizure prediction from isolated 4 s raw EEG windows. More informative experiments should evaluate patient-specific or seizure-specific horizons, longer temporal context, stricter interictal selection, and clinically meaningful alarm-level metrics.
