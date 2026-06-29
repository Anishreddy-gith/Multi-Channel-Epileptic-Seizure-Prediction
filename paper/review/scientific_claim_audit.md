# Scientific Claim Audit

This audit reviews the final manuscript claims against the frozen repository artifacts. Revisions use conservative wording and do not introduce new metrics.

| Claim | Status | Evidence | Final wording principle |
| --- | --- | --- | --- |
| The study is repository-backed and reproducible. | Partially supported | Paper package, logs, configs, checkpoints, and metrics exist; CUDA bitwise determinism is not guaranteed. | Use "repository-backed" and "reproducibility artifacts," not unconditional reproducibility. |
| The dataset has 10 patients, 273 EDFs, 50 seizures, and 734,796 windows. | Supported | `paper/tables/dataset_summary.csv`, label audit JSON. | Report exact measured counts. |
| The patient-wise split prevents patient identity leakage. | Supported for patient IDs | `paper/tables/patient_split_table.csv`, data audit outputs. | State no patient appears in more than one split; do not claim this solves all leakage risks. |
| The label audit found no label contamination. | Partially supported | Label audit counts are zero for implemented contamination checks. | Say "no evidence of mechanical contamination under the implemented checks." |
| The 10 min positive class is heterogeneous. | Partially supported | 47.26% of positives are more than 5 min before onset. | Present as concern or plausible explanation, not proof. |
| Imbalance handling did not solve the task. | Supported under this protocol | Imbalance-study metrics remain close to prevalence. | Say "did not produce strong ranking performance under this setup." |
| Attention and Hybrid models do not help. | Partially supported | Attention completed; Hybrid incomplete. | Say available evidence does not support increased capacity as sufficient; mark Hybrid incomplete. |
| The shorter-horizon hypothesis was rejected. | Too strong | Single split, no uncertainty estimates. | Say "not supported under the current experimental protocol." |
| Results are close to prevalence. | Supported qualitatively | Prevalence 1.9250%; final validation PR-AUC values around 0.0238--0.0258. | Quantify prevalence and avoid implying formal statistical equivalence. |
| The study has clinical implications. | Unsupported if stated strongly | No seizure-level/alarm-level metrics. | State the study is not clinically deployable and lacks alarm-level evaluation. |
| The pipeline is validated. | Too broad | Audits were internal. | Use "internally audited" or "frozen audited pipeline." |
| The models are state of the art. | Unsupported | Metrics are poor and no SOTA comparison is available. | Do not claim state of the art. |

## Strong Claims Removed or Weakened

- "validated CHB-MIT subset" -> "frozen dataset" or "internally audited subset"
- "label audit found no contamination" -> "label audit found no evidence of mechanical contamination under implemented checks"
- "horizon ablation rejected the hypothesis" -> "horizon ablation did not support the hypothesis under this protocol"
- "common deep models failed" -> "the evaluated models did not produce strong performance under this setup"
- "reproducible study" -> "repository-backed study with reproducibility artifacts"
