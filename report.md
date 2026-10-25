# SECOM yield root-cause report

**Data:** 1567 wafers, 19 Jul to 17 Oct 2008; 104 fails (6.6 %). 446 of 590 measurements usable after removing 28 mostly-missing and 116 constant ones.

## In plain words

Out of 446 measurements, **7** independent ones stand out in the pass/fail screening: **F059** (fails run high), **F103** (fails run high), **F247** (fails run high), **F477** (fails run high), **F129** (fails run high), **F028** (fails run low), **F033** (fails run high). 3 more move together with these and are probably the same causes seen by another sensor. These are the parameters to check first in the tool logs and maintenance records for the affected period. This is statistical evidence, not proof of cause; a controlled experiment (DOE) on the suspected parameters would confirm it.

A model trained on the first 70 % of production and tested on the last 30 % ranks wafers so that flagging the top 10 % catches **15 %** of fails (L1 logistic regression, PR-AUC 0.070 vs 0.055 for chance). That is close to chance: the patterns behind early failures did not carry forward, and the SPC charts show the process moving after the baseline period. Re-baselining the control limits and monitoring the suspects going forward is the practical next step.

## Top features by pass/fail screening

| Feature | p | q (FDR) | AUC | Fails run |
|---|---|---|---|---|
| F059 | 5.68e-11 | 0.000 | 0.692 | high |
| F103 | 9.55e-09 | 0.000 | 0.668 | high |
| F247 | 1.07e-06 | 0.000 | 0.675 | high |
| F519 | 1.19e-06 | 0.000 | 0.674 | high |
| F510 | 3.45e-06 | 0.000 | 0.636 | high |
| F477 | 3.50e-05 | 0.003 | 0.621 | high |
| F205 | 4.72e-05 | 0.003 | 0.619 | high |
| F129 | 4.83e-05 | 0.003 | 0.619 | high |
| F028 | 6.96e-05 | 0.003 | 0.383 | low |
| F033 | 1.61e-04 | 0.007 | 0.611 | high |
