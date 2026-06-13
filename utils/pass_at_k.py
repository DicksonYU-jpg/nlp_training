from __future__ import annotations

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al., 2021).

    n: total samples drawn, c: number correct, k: the k in pass@k.
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def compute_pass_at_k_for_dataset(
    per_problem_correct: list[int],
    n: int,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Average pass@k over all problems.

    per_problem_correct: number of correct samples (c) for each problem,
    each drawn from n total samples.
    """
    if k_values is None:
        k_values = [1, 8]
    results = {}
    for k in k_values:
        if k > n:
            raise ValueError(f"k={k} > n={n}")
        results[f"pass@{k}"] = float(
            np.mean([pass_at_k(n, c, k) for c in per_problem_correct])
        )
    return results
