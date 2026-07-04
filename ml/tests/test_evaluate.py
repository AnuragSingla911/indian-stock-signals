from iss.evaluate import EvalMetrics, evaluate


def test_evaluate_walk_forward_offline():
    m = evaluate(n_folds=4)
    assert isinstance(m, EvalMetrics)
    assert m.n_samples > 0
    assert 0 < m.n_test <= m.n_samples
    assert 0.0 <= m.base_rate <= 1.0
    assert 0.0 <= m.accuracy <= 1.0
    # AUC is a probability-like ranking metric bounded in [0, 1].
    assert 0.0 <= m.auc <= 1.0
    # Brier score for probabilities is bounded in [0, 1].
    assert 0.0 <= m.brier <= 1.0
    # Spearman IC is a correlation in [-1, 1].
    assert -1.0 <= m.information_coefficient <= 1.0
