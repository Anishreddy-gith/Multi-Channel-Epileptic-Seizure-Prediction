# Reviewer Question Audit

## Dataset and Split

**Why only 10 patients?**  
The completed artifact set contains 10 patients, 273 EDFs, and 50 seizures. The final paper reports this as a limitation. No additional patient-selection rationale was available in the frozen artifacts, so no unverified rationale was added.

**Why patient-wise splitting?**  
Patient-wise splitting prevents the same patient's EEG characteristics from appearing in both training and evaluation splits. The final paper states this explicitly and reports the exact patient IDs per split.

**Why no leave-one-patient-out validation?**  
LOPO results are unavailable. The final paper lists this as a limitation and future-work item.

**Are class distributions balanced across patients?**  
No. Positive ratios vary by patient from 0.7356% to 4.5038% in the stored class-distribution table. The final paper describes class and patient heterogeneity conservatively.

## Preprocessing and Labels

**Why 4 s windows and 2 s stride?**  
These are frozen completed-pipeline choices. The paper reports them and discusses 50% overlap and redundancy; it does not claim they are optimal.

**Why 10 min preictal horizon?**  
The 10 min horizon was the baseline completed label definition. The paper reports the later ablation over 7.5, 5, and 3 min horizons and states that shorter global horizons did not improve both validation PR-AUC and F1.

**Why 50 Hz notch filtering?**  
This was part of the frozen pipeline. Because CHB-MIT is a U.S. dataset where 60 Hz line noise may be relevant, the revised paper treats this as a limitation rather than a tuned choice.

**Could per-window z-score normalization remove predictive signal?**  
Possibly. The frozen artifacts do not test this. The paper reports the normalization and does not claim it is optimal.

## Models and Training

**Why EEGNet?**  
EEGNet is a compact established EEG neural baseline and has a low parameter count.

**Why Attention and Hybrid?**  
They test whether additional temporal/modeling capacity helps under the same data and split. The final paper avoids overinterpreting the incomplete Hybrid run.

**Why no classical baselines?**  
Classical baseline results are unavailable in the frozen paper package. The paper lists this as a limitation.

**Why no repeated seeds?**  
Repeated-seed results are unavailable. The final paper lists this as a statistical limitation.

## Evaluation

**Why PR-AUC?**  
The positive class prevalence is 1.9250%, so PR-AUC is more informative than accuracy for rare positives.

**Why no confidence intervals?**  
No confidence intervals or bootstrap estimates exist in the frozen artifacts. The paper states this explicitly.

**Why no seizure-level metrics or false alarms per hour?**  
They were not produced in the completed evaluation framework. The final paper makes clear that all reported metrics are window-level and that this limits clinical interpretation.

**Are raw PR-AUC values comparable across horizons with different prevalence?**  
Only cautiously. The paper now reports prevalence for each horizon and avoids claiming that horizon length is definitively irrelevant.
