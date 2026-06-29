# Reviewer Response Checklist

Use this checklist if preparing a rebuttal or revision.

## Critical

- Emphasize that the paper is an audited negative-result study, not a performance benchmark.
- State that no clinical deployment claim is made.
- Explain that all reported metrics come from frozen stored artifacts.
- Acknowledge that no confidence intervals, repeated seeds, or LOPO validation are available.
- Acknowledge that evaluation is window-level and lacks false alarms per hour.

## High

- Point reviewers to exact patient split IDs and class ratios.
- Explain why PR-AUC is primary under 1.9250% prevalence.
- Clarify that the label audit found no evidence of mechanical contamination, not that labels are physiologically perfect.
- Clarify that the horizon ablation tested only global horizons of 10, 7.5, 5, and 3 min.
- Clarify that Hybrid results are incomplete and included only for transparency.

## Medium

- Note that overlapping-window redundancy was measured and discussed.
- Note that threshold-dependent F1 is interpreted separately from PR-AUC.
- Note that model capacity was not sufficient under the tested setup.
- Note that the 50 Hz notch choice is a frozen-pipeline limitation.

## Low

- Keep figure and table captions descriptive rather than promotional.
- Avoid adding claims that are not traceable to CSV/JSON artifacts.
- Preserve exact metric values from stored files.
