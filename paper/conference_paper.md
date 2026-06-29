# A Patient-Wise CHB-MIT Seizure Prediction Study with Leakage, Label, and Horizon Audits

## Abstract

**Background:** Seizure prediction from scalp EEG is difficult to evaluate because patient leakage, class imbalance, overlapping windows, and broad preictal label definitions can produce misleading estimates of performance.  
**Objective:** This study reports a repository-backed, patient-wise CHB-MIT seizure prediction pipeline and uses the completed experiments to examine whether model choice, imbalance handling, or preictal horizon length explains poor discrimination.  
**Methods:** The frozen dataset contains 10 CHB-MIT patients, 273 EDF recordings, 50 annotated seizures, and 734,796 four-second windows. Each EDF was processed with 18 common EEG channels, 0.5--40 Hz bandpass filtering, 50 Hz notch filtering, 4 s windows with 2 s stride, per-window z-score normalization, a 10 min preictal definition, and a 60 s postictal exclusion interval. Training used a patient-wise split, AdamW, mixed precision, early stopping, checkpointing, and threshold optimization where specified. EEGNet, an attention model, and a hybrid EEGNet--Transformer were evaluated using PR-AUC, ROC-AUC, F1, MCC, sensitivity, and specificity.  
**Results:** The positive class prevalence was 1.9250%. In the final model comparison, EEGNet achieved validation PR-AUC 0.0239 and F1 0.0366, Attention achieved validation PR-AUC 0.0258 and F1 0.0000, and the time-limited Hybrid run achieved validation PR-AUC 0.0238 and F1 0.0461. The label audit found no evidence of mechanical label contamination under the implemented checks, but 6,685 of 14,145 positive windows, or 47.26%, occurred more than five minutes before seizure onset. A controlled preictal horizon ablation over 10, 7.5, 5, and 3 min did not produce a shorter horizon that improved both validation PR-AUC and validation F1 over the 10 min baseline.  
**Conclusion:** Under this frozen patient-wise protocol, the evaluated models and imbalance strategies produced PR-AUC values close to prevalence. These results do not support claims of clinical readiness or state-of-the-art performance. Instead, they provide an audited negative result showing that simple imbalance handling and global horizon shortening were insufficient for reliable patient-wise seizure prediction on this subset.

## 1. Introduction

Epileptic seizure prediction seeks to identify a preictal state before clinical seizure onset. Public EEG datasets such as CHB-MIT enable reproducible benchmarking, but the task is unusually sensitive to experimental design. Patient leakage can inflate performance, overlapping windows can reduce the effective number of independent samples, and the definition of "preictal" can convert a physiologically uncertain interval into a hard binary label.

This paper reports a completed and frozen CHB-MIT study whose main contribution is not a high-performing model, but an audited experimental account of what did and did not work under a patient-wise protocol. The study asks three practical questions:

1. Can standard imbalance strategies rescue near-prevalence PR-AUC under a patient-wise split?
2. Do attention or hybrid architectures improve discrimination over a compact EEGNet baseline?
3. Does shortening the preictal horizon improve validation PR-AUC and F1?

The answer to all three questions was negative under the completed protocol. We therefore frame the study as a reproducibility and problem-formulation contribution rather than a performance benchmark.

## 2. Related Work

CHB-MIT remains a common public scalp EEG benchmark for seizure detection and prediction, and PhysioNet provides the database infrastructure used by many biomedical signal processing studies. Prior seizure prediction work has emphasized that performance depends strongly on patient-specific versus patient-independent evaluation, seizure prediction horizon and seizure occurrence period definitions, feature representation, and preictal period selection. Reviews of seizure prediction have also warned that apparently strong results may not translate into clinically useful warning systems unless false alarm rates and temporal prediction constraints are evaluated.

EEGNet provides a compact convolutional architecture for EEG decoding and is a reasonable low-parameter neural baseline. Transformer-style attention models motivate testing whether longer-range temporal modeling can help, although the present study evaluates fixed 4 s windows rather than a full seizure-level temporal warning system. For class imbalance, PR-AUC is more informative than accuracy or ROC-AUC when positives are rare, and focal loss, weighted losses, and sampling strategies are standard options. These literatures motivate our emphasis on patient-wise splitting, PR-AUC, leakage audits, and conservative interpretation.

## 3. Dataset and Split

The processed subset contains 10 CHB-MIT patients, 273 EDF files, 50 seizure events, and 734,796 windows. The baseline 10 min label definition produced 14,145 positive windows and 720,651 negative windows, giving a positive ratio of 1.9250%.

The frozen patient-wise split was:

- Train: 439,738 windows from patients chb03, chb07, chb08, chb09, chb14, and chb23.
- Validation: 142,231 windows from patients chb01 and chb05.
- Test: 152,827 windows from patients chb02 and chb10.

No patient appears in more than one split. This is essential because window-level random splits can leak patient-specific EEG characteristics and seizure-proximity patterns into validation or test data.

## 4. Preprocessing

Each EDF was processed independently and stored as per-EDF shards. EDF files were checked for the 18 required common EEG channels; files missing required channels were skipped. Signals were filtered using a 0.5--40 Hz bandpass filter and a 50 Hz notch filter, segmented into 4 s windows with 2 s stride, and normalized independently per window and channel using z-score normalization.

The 50 Hz notch is reported as part of the frozen completed pipeline rather than as a tuned methodological claim. Because CHB-MIT is a U.S. dataset where 60 Hz line noise may be relevant, this choice is treated as a limitation and should be revisited in future experiments.

## 5. Label Generation and Audit

The primary binary label rule marks a window positive if it lies within 10 min before seizure onset. Seizure windows and the 60 s postictal exclusion interval are excluded from interictal negatives. Interictal windows before and after seizures are retained when they satisfy the exclusion rules.

The label audit found no evidence of mechanical contamination under the implemented checks:

- Positive windows outside implemented preictal intervals: 0.
- Positive windows overlapping seizure intervals: 0.
- Negative windows overlapping implemented preictal intervals: 0.
- Negative windows overlapping seizure intervals or postictal gaps: 0.

The main concern is therefore scientific label quality, not an observed coding mismatch. Of the 14,145 positive windows, 6,685 were more than 5 min before seizure onset. The median distance from positive-window end to seizure onset was 286 s. Because early and late preictal windows receive the same label, the positive class may be heterogeneous even when the implementation is internally consistent.

## 6. Models

Three neural models were evaluated:

- EEGNet, a compact convolutional EEG baseline with 26,770 parameters.
- An attention-based EEG model with 13,730 parameters.
- A hybrid EEGNet--Transformer with 295,906 parameters.

The Hybrid run was time-limited and is reported transparently as incomplete. Therefore, Hybrid results should be interpreted as an observed artifact of the completed run, not as a definitive architecture comparison.

## 7. Training Procedure

Training used the frozen sharded dataset and patient-wise split. The common training configuration was seed 42, batch size 128, AdamW with learning rate 0.0003 and weight decay 0.0001, ReduceLROnPlateau scheduling, automatic mixed precision on CUDA, gradient clipping, early stopping, and checkpointing. The focal-loss gamma was 2.0 when focal loss was used. DataLoader workers were set to 0 for Windows stability.

The imbalance study evaluated weighted cross entropy, focal loss, weighted random sampling, weighted cross entropy with threshold optimization, and focal loss with threshold optimization using EEGNet. Weighted cross entropy with threshold optimization was selected by validation PR-AUC with validation F1 as the tie-breaker.

## 8. Evaluation Metrics

The primary ranking metric is PR-AUC because positives are rare. ROC-AUC, accuracy, precision, recall, F1, balanced accuracy, MCC, Cohen's kappa, sensitivity, and specificity are also reported. F1 and threshold-dependent metrics depend on the selected threshold and should not be interpreted as pure ranking metrics.

This study reports window-level metrics only. It does not estimate seizure-level sensitivity, false alarms per hour, time in warning, or seizure prediction horizon/seizure occurrence period metrics. That omission limits clinical interpretation.

## 9. Results

### 9.1 Imbalance Study

**Observation:** The selected strategy was weighted cross entropy with threshold optimization. Its validation PR-AUC was 0.0239 and validation F1 was 0.0366. Focal loss variants increased threshold-dependent F1 in some cases but did not improve validation PR-AUC over the selected strategy. The weighted random sampler run was incomplete and is not used as a completed final strategy.

**Interpretation:** Imbalance handling changed operating points and threshold-dependent metrics, but it did not produce strong ranking performance. The results therefore do not support class imbalance alone as the dominant bottleneck.

### 9.2 Final Model Comparison

**Observation:** EEGNet achieved validation PR-AUC 0.0239, validation F1 0.0366, and test PR-AUC 0.0179. Attention achieved validation PR-AUC 0.0258 but F1 0.0000 at the selected threshold, with test PR-AUC 0.0163. The incomplete Hybrid run achieved validation PR-AUC 0.0238 and F1 0.0461, with test PR-AUC 0.0181.

**Interpretation:** None of the evaluated architectures produced validation PR-AUC substantially above the 1.9250% positive prevalence. The Attention model's higher validation PR-AUC did not translate into nonzero F1 under the selected operating point. The available evidence does not support increased model capacity as sufficient to solve the task under this setup.

### 9.3 Label Audit

**Observation:** The implemented labels passed contamination checks, but 47.26% of positive windows occurred more than 5 min before seizure onset. The median shared-overlap segment correlation between neighboring overlapping windows was 0.9895.

**Interpretation:** The poor model results are unlikely to be explained by obvious label-placement bugs. However, the positive class is broad, temporally heterogeneous, and highly redundant due to the 4 s window and 2 s stride design.

### 9.4 Preictal Horizon Ablation

**Observation:** The EEGNet horizon ablation evaluated 10, 7.5, 5, and 3 min. Validation PR-AUC values were 0.0239, 0.0211, 0.0140, and 0.0072, respectively. Validation F1 values were 0.0366, 0.0271, 0.0169, and 0.0076, respectively.

**Interpretation:** Shorter horizons reduced the number of positive windows but did not improve both validation PR-AUC and validation F1. Under this protocol, simple global horizon shortening was not sufficient to improve discrimination. This does not prove that preictal label design is irrelevant; it only shows that the tested global horizons did not improve the measured validation metrics.

## 10. Discussion

The central finding is a bounded negative result: under a patient-wise split and the completed preprocessing protocol, standard imbalance handling, threshold optimization, and the evaluated neural architectures did not produce reliable window-level seizure prediction.

The preprocessing and data audits matter because EEG seizure prediction pipelines are vulnerable to silent failure modes. If seizure windows, postictal windows, duplicate windows, or patient identities leak across splits, performance can appear artificially strong. The audit trail reduces this risk and makes the negative result more interpretable.

The label audit matters because poor performance across multiple model and loss configurations suggested that the bottleneck might be in the target definition. The audit found no evidence of mechanical label contamination, but it did show that nearly half of positive windows were more than five minutes before onset. This supports caution about broad fixed preictal labels, although the subsequent horizon ablation did not show that shortening the horizon alone improves performance.

The imbalance experiments matter because class prevalence is only 1.9250%. Accuracy is therefore uninformative, and even threshold-dependent metrics can be dominated by operating-point choices. PR-AUC and MCC indicate that the models remained close to prevalence-level ranking performance.

The negative findings are scientifically useful because they constrain future work. They suggest that simply adding capacity, reweighting losses, or shortening a global horizon may be insufficient for patient-wise CHB-MIT seizure prediction from isolated 4 s raw EEG windows. More informative experiments should evaluate patient-specific or seizure-specific horizons, longer temporal context, stricter interictal selection, and clinically meaningful alarm-level metrics.

## 11. Limitations

This study has important limitations:

- Only 10 CHB-MIT patients and 50 seizures were included in the completed artifact set.
- The evaluation uses a single patient-wise train/validation/test split rather than leave-one-patient-out or repeated patient-wise splits.
- No confidence intervals, bootstrap intervals, or repeated-seed variance estimates are available in the frozen experiments.
- Metrics are window-level rather than seizure-level or alarm-level.
- The 4 s windows with 2 s stride create 50% overlap and substantial redundancy.
- The models classify isolated windows rather than longer temporal sequences.
- The Hybrid model run was incomplete.
- The 50 Hz notch filter was part of the frozen pipeline but may not be optimal for CHB-MIT.
- The study does not evaluate classical feature-based baselines.
- The public dataset and subset selection limit generalization to other EEG montages, clinical settings, and patient populations.

These limitations should be read as constraints on inference, not as future-work items already solved by the present paper.

## 12. Future Work

Future work should evaluate leave-one-patient-out validation, repeated seeds, confidence intervals, seizure-level sensitivity, false alarms per hour, time-in-warning, explicit seizure prediction horizon and seizure occurrence period definitions, patient-specific preictal horizons, stricter interictal sampling, longer temporal context, and classical feature-based baselines. Revisiting line-noise filtering and evaluating 60 Hz notch filtering for CHB-MIT would also be appropriate.

## 13. Conclusion

This study provides a repository-backed patient-wise CHB-MIT seizure prediction experiment with leakage, label, imbalance, and horizon audits. The measured results do not support claims of strong predictive performance: validation PR-AUC remained close to positive prevalence, and no tested shorter preictal horizon improved both PR-AUC and F1 over the 10 min baseline. The main contribution is therefore an honest, reproducible negative finding that highlights the importance of label definition, patient-wise evaluation, and clinically meaningful metrics in seizure prediction research.

## References

References are provided in `paper/references/references.bib`.
