"""
Statistical tests for MedQA-USMLE evaluation.

Supports:
- McNemar's test (paired proportion comparison)
- Bootstrap 95% CI
- Cochran's Q (for >2 conditions)
- Wilcoxon signed-rank (for continuous metrics)
- Cohen's h / d effect sizes
"""

import numpy as np
from typing import Optional


def mcnemar_test(v3_correct: list[bool], v0_correct: list[bool]) -> dict:
    """
    McNemar's test for paired nominal data.
    
    Tests if the accuracy difference between V3 and V0 is significant.
    
    Contingency table:
                V0 right  V0 wrong
    V3 right      a          b
    V3 wrong      c          d
    
    H0: b = c (marginal homogeneity)
    Statistic: chi2 = (b - c)^2 / (b + c)
    
    Returns: dict with chi2, p_value, and contingency table
    """
    a = b = c = d = 0
    for v3, v0 in zip(v3_correct, v0_correct):
        if v3 and v0:
            a += 1
        elif v3 and not v0:
            b += 1
        elif not v3 and v0:
            c += 1
        else:
            d += 1

    n_discordant = b + c
    if n_discordant == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (b - c) ** 2 / n_discordant
        # Chi-squared with 1 degree of freedom
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(chi2, 1)

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "significant_05": p_value < 0.05,
        "contingency": {
            "both_correct": a,
            "v3_only_correct": b,
            "v0_only_correct": c,
            "both_wrong": d,
        },
        "effect_size": {
            "accuracy_gain": round((b - c) / (a + b + c + d), 4),
        }
    }


def bootstrap_ci(accuracies: list[float], n_iterations: int = 10000,
                 confidence: float = 0.95, seed: int = 42) -> dict:
    """
    Bootstrap confidence interval for accuracy.
    
    Args:
        accuracies: List of bool values (correct/incorrect per question)
        n_iterations: Number of bootstrap samples
        confidence: Confidence level (default 0.95)
        seed: Random seed
    
    Returns: dict with ci_lower, ci_upper, mean, std_error
    """
    rng = np.random.RandomState(seed)
    n = len(accuracies)
    scores = []
    for _ in range(n_iterations):
        sample = rng.choice(accuracies, size=n, replace=True)
        scores.append(np.mean(sample))
    scores = np.sort(scores)

    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(scores)),
        "ci_lower": float(round(lower, 4)),
        "ci_upper": float(round(upper, 4)),
        "std_error": float(round(np.std(scores), 4)),
        "n_iterations": n_iterations,
    }


def cohen_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for proportions.
    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    0.2 = small, 0.5 = medium, 0.8 = large
    """
    import math
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    return round(abs(h), 4)


def cohen_d(group1: list[float], group2: list[float]) -> float:
    """
    Cohen's d effect size for continuous measures.
    d = (mean1 - mean2) / pooled_std
    """
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return round(abs(m1 - m2) / pooled, 4)


def cochran_q(results_by_variant: dict[str, list[bool]]) -> dict:
    """
    Cochran's Q test for >2 related conditions.
    
    H0: All conditions have the same accuracy.
    
    Returns: dict with Q statistic, p_value, and pairwise comparisons
    """
    from scipy.stats import chi2
    
    variants = list(results_by_variant.keys())
    k = len(variants)
    n = len(results_by_variant[variants[0]])
    
    # Verify all variants have same length
    for v in variants:
        assert len(results_by_variant[v]) == n, f"Variant {v} has wrong length"
    
    # Row totals (correct count per question across all variants)
    row_totals = np.zeros(n)
    for v in variants:
        row_totals += np.array(results_by_variant[v], dtype=int)
    
    # Column totals (correct count per variant)
    col_totals = np.array([sum(results_by_variant[v]) for v in variants])
    
    grand_total = sum(col_totals)
    
    # Cochran's Q
    numerator = (k - 1) * (k * sum(col_totals**2) - grand_total**2)
    denominator = k * grand_total - sum(row_totals**2)
    
    q_stat = numerator / denominator if denominator != 0 else 0
    p_val = 1 - chi2.cdf(q_stat, k - 1)
    
    return {
        "Q_statistic": round(q_stat, 4),
        "p_value": round(p_val, 4),
        "significant_05": p_val < 0.05,
        "df": k - 1,
        "k": k,
        "n": n,
    }
