# Reviewer-Facing Checklist

## Evidence Boundaries

- [x] No state-of-the-art claim.
- [x] No clinical-readiness claim.
- [x] Hybrid marked incomplete.
- [x] Single split marked as a limitation.
- [x] Missing confidence intervals marked as unavailable.
- [x] Window-level evaluation distinguished from seizure-level evaluation.
- [x] Horizon ablation conclusion limited to the tested protocol.
- [x] Label audit described as an internal audit, not external validation.

## Likely Questions Answered

- [x] Why patient-wise split?
- [x] What patients are in each split?
- [x] What is the positive prevalence?
- [x] Why PR-AUC?
- [x] What exactly did the label audit find?
- [x] What did the horizon ablation test?
- [x] What did it not prove?
- [x] What are the main limitations?

## Remaining Vulnerabilities

- [ ] No LOPO validation.
- [ ] No repeated-seed variance.
- [ ] No confidence intervals.
- [ ] No alarm-level metrics.
- [ ] No classical feature baseline.
- [ ] Only 10 patients in the frozen artifact set.
