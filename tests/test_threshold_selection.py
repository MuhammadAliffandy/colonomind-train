import numpy as np
import pytest
from sklearn.model_selection import train_test_split
import src.train_dgx as train_dgx_module

from src.train_dgx import (
    build_arg_parser,
    oof_agent_predictions,
    select_threshold,
    validate_args,
)


def test_split_disjointness_three_way_indices():
    rng = np.random.default_rng(42)
    n = 500
    y = rng.integers(0, 3, size=n)
    idx_all = np.arange(n)

    val_es_frac = 0.15
    val_cal_frac = 0.15
    holdout_frac = val_es_frac + val_cal_frac

    idx_train, idx_holdout = train_test_split(
        idx_all,
        test_size=holdout_frac,
        random_state=42,
        stratify=y,
    )
    holdout_labels = y[idx_holdout]
    val_cal_ratio = val_cal_frac / holdout_frac
    idx_val_es, idx_val_cal = train_test_split(
        idx_holdout,
        test_size=val_cal_ratio,
        random_state=42,
        stratify=holdout_labels,
    )

    set_train = set(idx_train.tolist())
    set_val_es = set(idx_val_es.tolist())
    set_val_cal = set(idx_val_cal.tolist())

    assert set_train.isdisjoint(set_val_es)
    assert set_train.isdisjoint(set_val_cal)
    assert set_val_es.isdisjoint(set_val_cal)
    assert set_train | set_val_es | set_val_cal == set(idx_all.tolist())


def test_oof_agent_predictions_degrades_splits_and_is_deterministic():
    rng = np.random.default_rng(7)
    y = np.array([0] * 2 + [1] * 8 + [2] * 10)
    X = rng.normal(size=(len(y), 6))

    pred_1, splits_1 = oof_agent_predictions(X, y, seed=42, n_splits=5)
    pred_2, splits_2 = oof_agent_predictions(X, y, seed=42, n_splits=5)

    assert splits_1 == 2
    assert splits_2 == 2
    assert np.array_equal(pred_1, pred_2)


def test_select_threshold_purity_known_optimum(monkeypatch):
    monkeypatch.setattr(train_dgx_module, "_bootstrap_qwk_se", lambda *args, **kwargs: 0.0)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred_deep = np.array([0, 1, 1, 1, 2, 1])
    conf = np.array([0.20, 0.58, 0.62, 0.70, 0.90, 0.95])
    y_pred_agent_oof = np.array([0, 0, 1, 0, 2, 2])
    grid = np.array([0.30, 0.50, 0.60, 0.80])

    chosen, _, metadata = select_threshold(
        y_true=y_true,
        y_pred_deep=y_pred_deep,
        conf_deep=conf,
        y_pred_agent_oof=y_pred_agent_oof,
        grid=grid,
        seed=42,
    )

    assert chosen == pytest.approx(0.60)
    assert metadata["threshold_source"] == "tuned_val_cal"


def test_select_threshold_tie_break_lowest_threshold_within_one_se():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred_deep = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    conf = np.array([0.35, 0.36, 0.37, 0.38, 0.65, 0.66, 0.67, 0.68])
    y_pred_agent_oof = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    grid = np.array([0.30, 0.40, 0.50, 0.70])

    chosen, sweep, metadata = select_threshold(
        y_true=y_true,
        y_pred_deep=y_pred_deep,
        conf_deep=conf,
        y_pred_agent_oof=y_pred_agent_oof,
        grid=grid,
        seed=42,
    )

    best_qwk = max(row["qwk"] for row in sweep)
    cutoff = best_qwk - metadata["qwk_se_bootstrap"]
    within_one_se = [row["threshold"] for row in sweep if row["qwk"] >= cutoff]
    assert chosen == pytest.approx(min(within_one_se))


def test_select_threshold_degenerate_no_benefit():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred_deep = y_true.copy()
    conf = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    y_pred_agent_oof = np.array([1, 2, 0, 1, 2, 0])
    grid = np.array([0.30, 0.50, 0.70, 0.90])

    chosen, _, metadata = select_threshold(
        y_true=y_true,
        y_pred_deep=y_pred_deep,
        conf_deep=conf,
        y_pred_agent_oof=y_pred_agent_oof,
        grid=grid,
        seed=42,
    )

    assert chosen == pytest.approx(0.0)
    assert metadata["threshold_source"] == "degenerate_no_benefit"


def test_cli_requires_threshold_when_no_tune_threshold():
    parser = build_arg_parser(["resnet50"])
    args = parser.parse_args([
        "--scenario", "Intra",
        "--train_dataset", "NTUH",
        "--test_dataset", "LIMUC",
        "--model", "resnet50",
        "--no_tune_threshold",
    ])

    with pytest.raises(SystemExit):
        validate_args(args, parser)


def test_cli_override_allows_fixed_threshold_when_no_tune_threshold():
    parser = build_arg_parser(["resnet50"])
    args = parser.parse_args([
        "--scenario", "Intra",
        "--train_dataset", "NTUH",
        "--test_dataset", "LIMUC",
        "--model", "resnet50",
        "--no_tune_threshold",
        "--threshold", "0.7",
    ])

    validate_args(args, parser)
    assert args.tune_threshold is False
    assert args.threshold == pytest.approx(0.7)
