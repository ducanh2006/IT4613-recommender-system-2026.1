from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    mae,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)
from evaluation.stats import (
    bootstrap_ci,
    cliffs_delta,
    holm_bonferroni,
    paired_bootstrap_delta,
    summarize_values,
)


def test_rating_metrics():
    actual = [1.0, 2.0, 3.0]
    predicted = [1.5, 1.5, 3.5]
    assert rmse(actual, predicted) == pytest.approx(math.sqrt((0.25 + 0.25 + 0.25) / 3))
    assert mae(actual, predicted) == pytest.approx(0.5)


def test_ranking_metrics():
    recommended = [1, 2, 3]
    relevant = {1, 3, 5}
    assert precision_at_k(recommended, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(recommended, relevant, 3) == pytest.approx(2 / 3)
    assert hit_rate_at_k(recommended, relevant, 3) == 1.0
    assert mrr_at_k([2, 1], {1}, 2) == pytest.approx(0.5)
    assert average_precision_at_k([1, 3], relevant, 3) == pytest.approx(2 / 3)


def test_ndcg():
    relevance = {1: 2.0, 2: 0.0, 3: 1.0}
    assert ndcg_at_k([1, 3, 2], relevance, 3) == pytest.approx(1.0)
    assert ndcg_at_k([2, 3, 1], relevance, 3) < 1.0


def test_bootstrap_and_effect_size_are_deterministic():
    values = [0.1, 0.2, 0.3, 0.4]
    first = bootstrap_ci(values, n_boot=100, seed=7)
    second = bootstrap_ci(values, n_boot=100, seed=7)
    assert first == second
    delta = paired_bootstrap_delta(values, [0.0, 0.0, 0.0, 0.0], n_boot=100, seed=7)
    assert delta[0] == pytest.approx(0.25)
    assert cliffs_delta([2, 2, 2], [1, 1, 1]) == 1.0
    summary = summarize_values(values, n_boot=100, seed=7)
    assert summary["n"] == 4
    assert summary["mean"] == pytest.approx(np.mean(values))


def test_holm_bonferroni():
    adjusted = holm_bonferroni([0.01, 0.04, 0.03])
    assert adjusted[0] <= adjusted[2] <= 1.0
    assert all(0.0 <= value <= 1.0 for value in adjusted)
