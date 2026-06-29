# Revised Limitations

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

These limitations are constraints on inference, not completed future-work items.
