"""The outcome labels that cross module boundaries.

`records` **writes** the `outcome` column of the per-object dump; `crosscut`
**reads** it back. They used to declare these three strings separately, which
meant a rename in one place would not fail anywhere -- `cross_cut` would simply
build cells that matched nothing and report empty counts, which reads as "the
detector found nothing" rather than as an error.

Deliberately a leaf: this module imports nothing, so anything may import it
without pulling a dependency chain along. That is what lets a producer outside
`src/eval/` share the vocabulary of the files it writes.
"""

from __future__ import annotations

TP = "tp"   # a prediction that claimed a target
FP = "fp"   # a prediction that claimed nothing -- a false alarm
FN = "fn"   # a target no prediction claimed
