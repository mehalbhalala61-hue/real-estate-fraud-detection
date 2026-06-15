"""
src/models.py — Real Estate Fraud Detection
Model training — Logistic Regression, LightGBM, Stacking Ensemble.

CV Strategy: GroupKFold on city — prevents geographic leakage across folds.
MLflow: every run logged automatically — params, metrics, artifacts.

LEAKAGE RULES:
  - FeatureEngineer.fit() inside each fold on fold-train only
  - preprocessor.fit() inside each fold on fold-train only
  - city_fraud_rate (target encoding) computed OOF inside fold loop
  - Test set never touched during training
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Platt Calibration wrapper — module-level so pickle works
# ─────────────────────────────────────────────────────────────────────────────

class PlattCalibrated:
    """
    Wrapper that applies Platt scaling (sigmoid) on top of any base model.
    Must be module-level for pickle serialization to work.
    """
    def __init__(self, base_model, platt_lr):
        self.base_model = base_model
        self.platt_lr   = platt_lr

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.platt_lr.predict_proba(raw.reshape(-1, 1))
        return cal  # shape (n, 2) — col 0 = P(normal), col 1 = P(fraud)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_mlflow(cfg: dict) -> None:
    """Set tracking URI and experiment — call once at notebook start."""
    # Silence git warning — git not needed for local MLflow tracking
    import os
    os.environ.setdefault('GIT_PYTHON_REFRESH', 'quiet')

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    logger.info(
        f"MLflow: tracking_uri={cfg['mlflow']['tracking_uri']} | "
        f"experiment={cfg['mlflow']['experiment_name']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers — imported from evaluate.py inside functions to avoid circulars
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fold_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """PR-AUC + Recall@95Precision for one fold."""
    from src.evaluate import pr_auc_score, recall_at_precision
    return {
        "pr_auc": pr_auc_score(y_true, y_prob),
        "recall_at_95p": recall_at_precision(y_true, y_prob, target_precision=0.95),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core CV loop — reused by all model tiers
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    build_model_fn,          # callable() → unfitted sklearn estimator
    run_name: str,
    extra_params: dict = {},
    use_text: bool = False,
) -> Dict:
    """
    GroupKFold CV with full feature engineering inside each fold.
    Logs every fold + mean metrics to MLflow.

    Returns:
        {
          'oof_probs':    np.ndarray of shape (n_samples,),
          'fold_metrics': list of per-fold dicts,
          'mean_pr_auc':  float,
          'mean_recall':  float,
          'std_pr_auc':   float,
          'run_id':       str,
        }
    """
    from src.features import FeatureEngineer
    from src.preprocessing import build_preprocessor, get_feature_names, get_feature_names
    from src.text_features import TextPipeline, is_text_enabled

    n_folds    = cfg["model"]["cv_folds"]
    group_col  = cfg["columns"]["group_col"]       # city
    text_on    = use_text and is_text_enabled(cfg)

    # FIX: city column mein None/NaN values hain — numpy sort crash karta tha
    # fillna ensures GroupKFold can always sort groups
    groups     = X[group_col].fillna('__unknown__').values
    gkf        = GroupKFold(n_splits=n_folds)

    oof_probs    = np.zeros(len(X))
    fold_metrics = []

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "model": run_name,
            "n_folds": n_folds,
            "cv_strategy": "GroupKFold(city)",
            "use_text": text_on,
            **extra_params,
        })

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            fold_start = time.time()

            X_fold_tr = X.iloc[train_idx].copy()
            X_fold_val = X.iloc[val_idx].copy()
            y_fold_tr = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]

            # ── Feature engineering — fit on fold-train only ──────────────
            feat_eng = FeatureEngineer(cfg)
            feat_eng.fit(X_fold_tr, y=y_fold_tr)   # city_fraud_rate OOF-safe
            X_fold_tr  = feat_eng.transform(X_fold_tr)
            X_fold_val = feat_eng.transform(X_fold_val)

            # ── Stateless features ────────────────────────────────────────
            X_fold_tr  = FeatureEngineer.add_stateless_features(X_fold_tr, cfg)
            X_fold_val = FeatureEngineer.add_stateless_features(X_fold_val, cfg)

            # ── Preprocessing — fit on fold-train only ────────────────────
            preprocessor = build_preprocessor(cfg, include_city_fraud_rate=True)
            preprocessor.fit(X_fold_tr)
            X_tr_proc  = preprocessor.transform(X_fold_tr)
            X_val_proc = preprocessor.transform(X_fold_val)

            # ── Optional text features ────────────────────────────────────
            if text_on and cfg["text_features"]["column"] in X.columns:
                text_pipe = TextPipeline(cfg)
                text_tr   = text_pipe.fit_transform(X_fold_tr[cfg["text_features"]["column"]])
                text_val  = text_pipe.transform(X_fold_val[cfg["text_features"]["column"]])
                X_tr_proc  = np.hstack([X_tr_proc, text_tr])
                X_val_proc = np.hstack([X_val_proc, text_val])

            # ── Train + predict ───────────────────────────────────────────
            model = build_model_fn()
            model.fit(X_tr_proc, y_fold_tr)
            val_probs = model.predict_proba(X_val_proc)[:, 1]
            oof_probs[val_idx] = val_probs

            # ── Fold metrics ──────────────────────────────────────────────
            metrics = _compute_fold_metrics(y_fold_val.values, val_probs)
            fold_metrics.append(metrics)
            elapsed = time.time() - fold_start

            mlflow.log_metrics({
                f"fold_{fold_idx+1}_pr_auc":       metrics["pr_auc"],
                f"fold_{fold_idx+1}_recall_at_95p": metrics["recall_at_95p"],
            })
            logger.info(
                f"  Fold {fold_idx+1}/{n_folds} — "
                f"PR-AUC: {metrics['pr_auc']:.4f} | "
                f"Recall@95P: {metrics['recall_at_95p']:.4f} | "
                f"{elapsed:.1f}s"
            )

        # ── Aggregate metrics ─────────────────────────────────────────────
        pr_aucs  = [m["pr_auc"]        for m in fold_metrics]
        recalls  = [m["recall_at_95p"] for m in fold_metrics]
        mean_pr  = float(np.mean(pr_aucs))
        std_pr   = float(np.std(pr_aucs))
        mean_rec = float(np.mean(recalls))

        mlflow.log_metrics({
            "mean_pr_auc":       mean_pr,
            "std_pr_auc":        std_pr,
            "mean_recall_at_95p": mean_rec,
        })

        # OOF overall
        from src.evaluate import pr_auc_score, recall_at_precision
        oof_pr_auc = pr_auc_score(y.values, oof_probs)
        mlflow.log_metric("oof_pr_auc", oof_pr_auc)

        logger.info(
            f"\n{'='*55}\n"
            f"  {run_name} CV Results\n"
            f"  PR-AUC  : {mean_pr:.4f} ± {std_pr:.4f}\n"
            f"  Recall@95P: {mean_rec:.4f}\n"
            f"  OOF PR-AUC: {oof_pr_auc:.4f}\n"
            f"{'='*55}"
        )

        run_id = run.info.run_id

    return {
        "oof_probs":    oof_probs,
        "fold_metrics": fold_metrics,
        "mean_pr_auc":  mean_pr,
        "std_pr_auc":   std_pr,
        "mean_recall":  mean_rec,
        "oof_pr_auc":   oof_pr_auc,
        "run_id":       run_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Logistic Regression baseline
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    C: float = 1.0,
) -> Dict:
    """
    Tier 1: Logistic Regression (L2) — linear baseline.
    Sets the improvement floor for LightGBM.
    """
    logger.info("Training Tier 1 — Logistic Regression (baseline)")

    def build():
        return LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",   # handles imbalance
            solver="lbfgs",
            random_state=cfg["model"]["random_state"],
            n_jobs=-1,
        )

    return run_cv(
        X, y, cfg,
        build_model_fn=build,
        run_name="baseline_lr",
        extra_params={"C": C, "class_weight": "balanced"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — LightGBM tabular baseline
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm_tabular(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Tier 2: LightGBM on tabular features only.
    scale_pos_weight handles class imbalance automatically.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")

    logger.info("Training Tier 2 — LightGBM tabular baseline")

    lgbm_cfg = cfg["model"]["lgbm_defaults"].copy()
    if params:
        lgbm_cfg.update(params)

    # Auto scale_pos_weight
    fraud_rate = float(y.mean())
    scale_pos  = round((1 - fraud_rate) / fraud_rate, 2)
    lgbm_cfg["scale_pos_weight"] = scale_pos
    logger.info(f"  scale_pos_weight: {scale_pos:.2f} (fraud_rate={fraud_rate:.3f})")

    def build():
        p = lgbm_cfg.copy()
        p.pop("scale_pos_weight", None)   # set via constructor
        return lgb.LGBMClassifier(
            scale_pos_weight=scale_pos,
            **p,
        )

    return run_cv(
        X, y, cfg,
        build_model_fn=build,
        run_name="baseline_lgbm",
        extra_params={k: v for k, v in lgbm_cfg.items()},
        use_text=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — LightGBM tabular + text
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm_text(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Tier 3: LightGBM with tabular + TF-IDF/SVD text features.
    Returns zeros for text when text.enabled=False (current dataset).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")

    logger.info("Training Tier 3 — LightGBM tabular + text")

    lgbm_cfg = cfg["model"]["lgbm_defaults"].copy()
    if params:
        lgbm_cfg.update(params)

    fraud_rate = float(y.mean())
    scale_pos  = round((1 - fraud_rate) / fraud_rate, 2)
    lgbm_cfg["scale_pos_weight"] = scale_pos

    def build():
        p = lgbm_cfg.copy()
        p.pop("scale_pos_weight", None)
        return lgb.LGBMClassifier(scale_pos_weight=scale_pos, **p)

    return run_cv(
        X, y, cfg,
        build_model_fn=build,
        run_name="baseline_lgbm_text",
        extra_params={k: v for k, v in lgbm_cfg.items()},
        use_text=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stacking ensemble — Day 8
# ─────────────────────────────────────────────────────────────────────────────

def train_stacking_meta(
    oof_predictions: Dict[str, np.ndarray],
    y: pd.Series,
    cfg: dict,
) -> Tuple:
    """
    Day 8: Train meta-model on OOF predictions from base models.

    oof_predictions: {'lr': array, 'lgbm': array, 'lgbm_text': array}
    Returns: (fitted_meta_model, stacking_train_df)

    CRITICAL: Only OOF predictions — NO raw features in stacking input.
    Raw features = overfitting on stacking layer.
    """
    logger.info("Training stacking meta-model on OOF predictions")

    # Build stacking matrix — shape (n_samples, n_base_models)
    stacking_df = pd.DataFrame(oof_predictions)
    logger.info(f"Stacking input shape: {stacking_df.shape}")
    logger.info(f"Columns: {list(stacking_df.columns)}")

    # Meta-model: simple LR with regularization
    meta_C = cfg["stacking"].get("meta_model_C", 0.1)
    meta_model = LogisticRegression(
        C=meta_C,
        max_iter=1000,
        random_state=cfg["model"]["random_state"],
    )
    meta_model.fit(stacking_df.values, y.values)

    from src.evaluate import pr_auc_score
    meta_probs  = meta_model.predict_proba(stacking_df.values)[:, 1]
    meta_pr_auc = pr_auc_score(y.values, meta_probs)
    logger.info(f"Meta-model OOF PR-AUC: {meta_pr_auc:.4f}")

    with mlflow.start_run(run_name="stacking_meta"):
        mlflow.log_params({"meta_model": "logistic_regression", "C": meta_C})
        mlflow.log_metric("meta_oof_pr_auc", meta_pr_auc)
        mlflow.sklearn.log_model(meta_model, "meta_model")

    return meta_model, stacking_df


# ─────────────────────────────────────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved → {path}")


def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded ← {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Baseline report builder
# ─────────────────────────────────────────────────────────────────────────────

def build_baseline_report(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build comparison table from run_cv() results dict.
    results: {'Logistic Regression': {...}, 'LightGBM': {...}, ...}
    """
    rows = []
    for model_name, res in results.items():
        rows.append({
            "Model":        model_name,
            "PR-AUC Mean":  round(res["mean_pr_auc"], 4),
            "PR-AUC Std":   round(res["std_pr_auc"],  4),
            "Recall@95P":   round(res["mean_recall"],  4),
            "OOF PR-AUC":   round(res["oof_pr_auc"],   4),
            "MLflow Run ID": res["run_id"][:8] + "...",
        })
    return pd.DataFrame(rows).sort_values("PR-AUC Mean", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Optuna tuning — Day 6
# ─────────────────────────────────────────────────────────────────────────────

def tune_lgbm_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    completed_folds: int = 0,
    outer_scores_so_far: Optional[list] = None,
    best_params_so_far: Optional[list] = None,
) -> Dict:
    """
    Nested CV Optuna tuning for LightGBM.

    Nested CV structure:
      - Outer loop : GroupKFold(5) — honest evaluation, test never touched
      - Inner loop : GroupKFold(3) — Optuna optimises on this
      - Test set   : NEVER used during tuning

    Returns best_params dict + study object.
    Saves best params to configs/best_params.yaml automatically.
    """
    try:
        import optuna
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install optuna lightgbm")

    import yaml
    from src.features import FeatureEngineer
    from src.preprocessing import build_preprocessor, get_feature_names

    # Silence Optuna logs — MLflow handles run tracking
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    group_col  = cfg["columns"]["group_col"]
    groups     = X[group_col].fillna("__unknown__").values
    n_folds    = cfg["model"]["cv_folds"]
    gkf_outer  = GroupKFold(n_splits=n_folds)

    fraud_rate = float(y.mean())
    scale_pos  = round((1 - fraud_rate) / fraud_rate, 2)

    def _build_features(X_tr, X_val, y_tr):
        """Feature engineering inside a fold — fit on train only."""
        # Silence inner-fold logging — 750 fits would spam the output
        import logging as _log
        _log.getLogger('src.features').setLevel(_log.WARNING)
        _log.getLogger('src.preprocessing').setLevel(_log.WARNING)

        fe = FeatureEngineer(cfg)
        fe.fit(X_tr, y=y_tr)
        X_tr  = fe.transform(X_tr)
        X_val = fe.transform(X_val)
        X_tr  = FeatureEngineer.add_stateless_features(X_tr, cfg)
        X_val = FeatureEngineer.add_stateless_features(X_val, cfg)
        pre = build_preprocessor(cfg, include_city_fraud_rate=True)
        pre.fit(X_tr)

        # FIX: return DataFrame with feature names — prevents LightGBM UserWarning
        # "X does not have valid feature names" fires when preprocessor is fit on
        # DataFrame but returns numpy array — LightGBM sees inconsistency
        feat_names = get_feature_names(cfg, include_city_fraud_rate=True)
        X_tr_out  = pd.DataFrame(pre.transform(X_tr),  columns=feat_names)
        X_val_out = pd.DataFrame(pre.transform(X_val), columns=feat_names)

        # Restore logging level for outer-fold messages
        _log.getLogger('src.features').setLevel(_log.INFO)
        _log.getLogger('src.preprocessing').setLevel(_log.INFO)

        return X_tr_out, X_val_out

    def _objective(trial, X_inner, y_inner, groups_inner):
        """Optuna objective — inner CV PR-AUC."""
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":         trial.suggest_int("max_depth", 4, 10),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight":  scale_pos,
            "random_state":      cfg["model"]["random_state"],
            "n_jobs":            -1,
            "verbose":           -1,
        }

        inner_gkf = GroupKFold(n_splits=3)
        fold_scores = []

        for tr_idx, val_idx in inner_gkf.split(X_inner, y_inner, groups_inner):
            X_tr  = X_inner.iloc[tr_idx]
            X_val = X_inner.iloc[val_idx]
            y_tr  = y_inner.iloc[tr_idx]
            y_val = y_inner.iloc[val_idx]

            X_tr_p, X_val_p = _build_features(X_tr, X_val, y_tr)

            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr_p, y_tr)
            probs = model.predict_proba(X_val_p)[:, 1]

            from src.evaluate import pr_auc_score
            fold_scores.append(pr_auc_score(y_val.values, probs))

        return float(np.mean(fold_scores))

    # ── Outer CV — checkpoint resume support ─────────────────────────────
    outer_scores      = list(outer_scores_so_far or [])
    best_params_per_fold = list(best_params_so_far or [])

    with mlflow.start_run(run_name="optuna_tuning") as run:
        mlflow.log_params({
            "n_trials":      n_trials,
            "cv_outer":      n_folds,
            "cv_inner":      3,
            "cv_strategy":   "nested_GroupKFold(city)",
            "scale_pos_weight": scale_pos,
        })

        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf_outer.split(X, y, groups)
        ):
            # Skip already-completed folds (checkpoint resume)
            if fold_idx < completed_folds:
                logger.info(f"Outer fold {fold_idx+1}/{n_folds} — skipping (checkpoint)")
                continue

            logger.info(f"Outer fold {fold_idx+1}/{n_folds} — running {n_trials} Optuna trials")

            X_outer_tr  = X.iloc[train_idx]
            X_outer_val = X.iloc[val_idx]
            y_outer_tr  = y.iloc[train_idx]
            y_outer_val = y.iloc[val_idx]
            groups_inner = X_outer_tr[group_col].fillna("__unknown__").values

            study = optuna.create_study(direction="maximize")

            # Manual progress callback — works without ipywidgets
            completed = [0]
            def _progress_cb(study, trial):
                completed[0] += 1
                if completed[0] % 10 == 0 or completed[0] == n_trials:
                    print(f"    Fold {fold_idx+1} trial {completed[0]}/{n_trials} — best PR-AUC: {study.best_value:.4f}", flush=True)

            study.optimize(
                lambda trial: _objective(trial, X_outer_tr, y_outer_tr, groups_inner),
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False,
                callbacks=[_progress_cb],
            )

            best_p = study.best_params
            best_params_per_fold.append(best_p)

            # Evaluate best params on outer val fold
            X_tr_p, X_val_p = _build_features(X_outer_tr, X_outer_val, y_outer_tr)
            model = lgb.LGBMClassifier(
                **best_p,
                scale_pos_weight=scale_pos,
                random_state=cfg["model"]["random_state"],
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_tr_p, y_outer_tr)
            probs = model.predict_proba(X_val_p)[:, 1]

            from src.evaluate import pr_auc_score
            fold_pr = pr_auc_score(y_outer_val.values, probs)
            outer_scores.append(fold_pr)

            mlflow.log_metrics({
                f"outer_fold_{fold_idx+1}_pr_auc":    fold_pr,
                f"outer_fold_{fold_idx+1}_best_value": study.best_value,
            })
            logger.info(
                f"  Fold {fold_idx+1} — inner best: {study.best_value:.4f} | "
                f"outer val: {fold_pr:.4f}"
            )

            # Save checkpoint after every fold — safe against kernel/crash
            if checkpoint_path:
                import pickle as _pkl
                _ckpt = {
                    "completed_folds":     fold_idx + 1,
                    "outer_scores":        outer_scores,
                    "best_params_per_fold": best_params_per_fold,
                }
                with open(checkpoint_path, "wb") as _f:
                    _pkl.dump(_ckpt, _f)
                logger.info(f"  ✅ Checkpoint saved → {checkpoint_path} (fold {fold_idx+1}/{n_folds} done)")

        mean_outer = float(np.mean(outer_scores))
        std_outer  = float(np.std(outer_scores))
        mlflow.log_metrics({"mean_outer_pr_auc": mean_outer, "std_outer_pr_auc": std_outer})
        run_id = run.info.run_id

    # Best params = params from fold with highest outer score
    best_fold_idx  = int(np.argmax(outer_scores))
    best_params    = best_params_per_fold[best_fold_idx]

    # Add fixed params that aren't tuned
    best_params["scale_pos_weight"] = scale_pos
    best_params["random_state"]     = cfg["model"]["random_state"]
    best_params["n_jobs"]           = -1
    best_params["verbose"]          = -1

    # Save to configs/best_params.yaml
    best_params_path = Path("configs/best_params.yaml")
    with open(best_params_path, "w") as f:
        yaml.dump({"lgbm_best_params": best_params}, f, default_flow_style=False)
    logger.info(f"Best params saved → {best_params_path}")

    logger.info(
        f"\n{'='*55}\n"
        f"  Optuna Tuning Complete\n"
        f"  Outer CV PR-AUC: {mean_outer:.4f} ± {std_outer:.4f}\n"
        f"  Best fold: {best_fold_idx+1} (PR-AUC={outer_scores[best_fold_idx]:.4f})\n"
        f"  Best params: {best_params}\n"
        f"{'='*55}"
    )

    return {
        "best_params":         best_params,
        "outer_scores":        outer_scores,
        "mean_outer_pr_auc":   mean_outer,
        "std_outer_pr_auc":    std_outer,
        "best_params_per_fold": best_params_per_fold,
        "run_id":              run_id,
    }


def retrain_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    best_params: Dict,
) -> Dict:
    """
    Day 7: Retrain LightGBM with best Optuna params.
    Full GroupKFold CV — new OOF predictions saved.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")

    logger.info("Day 7 — Retraining with best Optuna params")

    params = best_params.copy()

    def build():
        p = {k: v for k, v in params.items() if k != "scale_pos_weight"}
        return lgb.LGBMClassifier(
            scale_pos_weight=params.get("scale_pos_weight", 1.0),
            **p,
        )

    return run_cv(
        X, y, cfg,
        build_model_fn=build,
        run_name="lgbm_tuned",
        extra_params=params,
        use_text=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Probability Calibration — Day 8
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_model(
    meta_model,
    stacking_df: "pd.DataFrame",
    y: "pd.Series",
    cfg: dict,
    method: str = "sigmoid",
) -> Tuple:
    """
    Platt scaling (sigmoid) calibration on stacking OOF predictions.

    Why calibrate?
      Raw model scores are not probabilities — score=0.8 doesn't mean
      80% chance of fraud. Calibration ensures p=0.8 means ~80% fraud.
      Required for threshold_decisions.md to be meaningful.

    Returns: (calibrated_model, calibration_report_dict)
    """
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    logger.info(f"Calibrating meta-model — method: {method}")

    # FIX: cv="prefit" removed in sklearn >= 1.2
    # Platt scaling manually — fit LogisticRegression on raw OOF probabilities
    raw_probs = meta_model.predict_proba(stacking_df.values)[:, 1]

    platt_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg["model"]["random_state"])
    platt_lr.fit(raw_probs.reshape(-1, 1), y.values)

    calibrated = PlattCalibrated(meta_model, platt_lr)
    cal_probs  = calibrated.predict_proba(stacking_df.values)[:, 1]

    # Calibration curve — how close to diagonal?
    prob_true_raw, prob_pred_raw = calibration_curve(y.values, raw_probs,  n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y.values, cal_probs,  n_bins=10)

    # Mean absolute calibration error
    mace_raw = float(np.mean(np.abs(prob_true_raw - prob_pred_raw)))
    mace_cal = float(np.mean(np.abs(prob_true_cal - prob_pred_cal)))

    from src.evaluate import pr_auc_score
    cal_pr_auc = pr_auc_score(y.values, cal_probs)

    report = {
        "method":        method,
        "mace_before":   round(mace_raw, 4),
        "mace_after":    round(mace_cal, 4),
        "mace_improved": mace_cal < mace_raw,
        "cal_pr_auc":    round(cal_pr_auc, 4),
    }

    logger.info(
        f"  Calibration MACE: {mace_raw:.4f} → {mace_cal:.4f} "
        f"({'improved' if mace_cal < mace_raw else 'no improvement'})"
    )
    logger.info(f"  Calibrated PR-AUC: {cal_pr_auc:.4f}")

    # Save calibrated model
    cal_path = cfg["paths"]["calibrated_model"]
    save_model(calibrated, cal_path)
    logger.info(f"  Calibrated model saved → {cal_path}")

    # Log to MLflow
    with mlflow.start_run(run_name="calibration"):
        mlflow.log_params({"method": method})
        mlflow.log_metrics({
            "mace_before":  mace_raw,
            "mace_after":   mace_cal,
            "cal_pr_auc":   cal_pr_auc,
        })
        # _PlattCalibrated is not sklearn estimator — log as pickle artifact
        import pickle as _pkl, tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            _pkl.dump(calibrated, tmp)
            mlflow.log_artifact(tmp.name, "calibrated_model")
        _os.unlink(tmp.name)

    return calibrated, report