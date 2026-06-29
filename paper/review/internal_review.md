# Internal Peer Review

## Scores

| Category | Score / 10 | Rationale |
| --- | ---: | --- |
| Novelty | 5 | The work is not a new model, but the combination of patient-wise training, leakage audit, label audit, imbalance study, and horizon ablation is useful as a reproducibility-focused negative result. |
| Technical quality | 6 | The pipeline artifacts are systematic and internally consistent, but the 50 Hz notch choice, incomplete Hybrid run, and absence of classical baselines weaken the technical package. |
| Methodology | 5 | Patient-wise evaluation is a strength. Single split, no confidence intervals, window-level metrics, and no seizure-level false-alarm analysis are major limitations. |
| Reproducibility | 8 | The package includes configs, logs, checkpoints, metrics, software versions, hardware, tables, and generated figures. Bitwise CUDA reproducibility is not guaranteed. |
| Writing | 7 | The revised manuscript is clearer and more conservative than the original. It still reads as a compact conference paper rather than a full clinical validation study. |
| Experimental design | 5 | The design is controlled for the completed hypotheses but lacks LOPO, repeated seeds, feature baselines, and alarm-level evaluation. |
| Statistical reporting | 3 | No confidence intervals, repeated seeds, or formal statistical tests are available. |
| Presentation | 7 | Tables, figures, captions, and limitations are now clearer. Wide raw tables remain available in the package but are not ideal for main-paper layout. |

## Overall Recommendation

**Borderline / Weak Reject for a full conference track; Weak Accept for a reproducibility, negative-results, or biomedical ML workshop track.**

The paper should be submitted only with conservative claims. Its value is in the audited negative result, not predictive performance.

## Highest-Risk Reviewer Objections

1. Window-level metrics do not establish clinical seizure prediction.
2. The 10-patient subset and single split limit generalizability.
3. No confidence intervals or repeated seeds are available.
4. Hybrid results are incomplete.
5. Horizon-ablation PR-AUC comparisons are affected by changing prevalence.
6. No classical feature-based baselines are reported.
7. The 50 Hz notch filter choice needs caution for CHB-MIT.

## Revision Actions Completed

- Replaced overly strong validation language with "internally audited" wording.
- Marked Hybrid as incomplete wherever model comparisons are discussed.
- Added prevalence context for PR-AUC.
- Added 50 Hz notch limitation.
- Separated observations from interpretations in the Results section.
- Added a stronger Limitations section.
- Added reviewer-question and claim-audit files.
