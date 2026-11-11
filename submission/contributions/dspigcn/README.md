# DSPIGCN contribution — pending formal evidence

Owner: YAN Zhengyang. Code and reproducible commands:
[`models/dspigcn`](../../../models/dspigcn/README.md).

Linux benchmark and 11 owner tests passed. The current A800 two-epoch ETH
pilot passed all six conditions (see `evidence/pilot_*`). Its epoch durations
were 4.305 and 3.631 seconds, with peak allocated GPU memory 503,714,304 bytes.
These pilot measurements are not formal accuracy results. The 15 formal fits
are running with the registered 200-epoch budget.

The report/deck sources use the shared metric macros; they become renderable
after the complete audited matrix and figure generation. No placeholder
numbers or paper-reported values are supplied as experimental results.

Completion sequence: Linux benchmark and tests → A800 pilot and audit →
five folds × three seeds × six conditions → common CPU score/audit → runtime
and diagnostics → figures and interpreted report → packaged artifact and
independent extraction verification → owner preview → editor review.

Local Mac smoke/pilot evidence is development evidence and must not be mixed
into the formal Linux run matrix. Keep the registry pending until review.
