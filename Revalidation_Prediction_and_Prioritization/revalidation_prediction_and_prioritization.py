"""CPIS 603 | BPI Challenge 2017

Six models: Logistic Regression, Decision Tree, Random Forest, XGBoost,
CatBoost, and Contextual Bandit (REINFORCE).

Pipeline: fresh XES parsing -> prediction dataset -> chronological training and validation
-> six-model comparison -> select best model -> fixed 10% risk-based prioritization.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lxml import etree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn


# 1. PROJECT SETTINGS
XES_FILE = None
OUTPUT_FOLDER = "CPIS603_Best_Model_Risk_Results"
REVIEW_CAPACITY = 0.10
RANDOM_REPETITIONS = 1000
BANDIT_MAX_EPOCHS = 65
BANDIT_HIDDEN_UNITS = 48
SAVE_FEATURE_IMPORTANCE = True
RANDOM_SEED = 42

PROJECT_FOLDER = Path(__file__).resolve().parent
CASE = "case:concept:name"
ACTIVITY = "concept:name"
TIME = "time:timestamp"
LIFECYCLE = "lifecycle:transition"
RESOURCE = "org:resource"
TARGET = "Repeated_Revalidation_Rework"
CAPACITIES = (REVIEW_CAPACITY,)
SEED = RANDOM_SEED

COUNT_FEATURES = {
    "W_Validate_Count_So_Far": "W_Validate application",
    "A_Validating_Count_So_Far": "A_Validating",
    "A_Incomplete_Count_So_Far": "A_Incomplete",
    "W_Call_Incomplete_Count_So_Far": "W_Call incomplete files",
    "W_Complete_Count_So_Far": "W_Complete application",
    "A_Complete_Count_So_Far": "A_Complete",
    "O_Create_Offer_Count_So_Far": "O_Create Offer",
    "O_Sent_Mail_Online_Count_So_Far": "O_Sent (mail and online)",
    "O_Sent_Online_Only_Count_So_Far": "O_Sent (online only)",
}
OPTIONAL_CASE_ATTRIBUTES = [
    "case:RequestedAmount", "case:LoanGoal", "case:ApplicationType",
]



# 2. BUILD THE PREDICTIVE DATASET

def _stamp(val: str) -> datetime:
    d = datetime.fromisoformat(val.replace('Z', '+00:00'))
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)


def _entry(element) -> tuple[str, str] | None:
    key = element.get('key')
    val = element.get('value')
    if key is None or val is None:
        return None
    return (key, val)


def _business_event(activity: str, lifecycle: str) -> bool:
    return activity.startswith(('A_', 'O_')) and lifecycle == 'complete' or (activity.startswith('W_') and lifecycle == 'schedule')


def parse_xes_and_build_dataset(path: Path, every: int=5000) -> tuple[pd.DataFrame, dict]:
    import gzip
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'XES not found: {path}')
    opener = gzip.open if path.name.lower().endswith('.gz') else open
    rows = []
    total_traces = raw_events = business_events = eligible = 0
    with opener(path, 'rb') as handle:
        context = etree.iterparse(handle, events=('end',), tag='trace', huge_tree=True)
        for _, trace in context:
            total_traces += 1
            trace_props = {}
            raw_event_elements = []
            for child in trace:
                if child.tag == 'event':
                    raw_event_elements.append(child)
                elif child.get('key') is not None:
                    ent = _entry(child)
                    if ent:
                        trace_props[ent[0]] = ent[1]
            case_id = trace_props.get('concept:name')
            if not case_id:
                raise ValueError(f'Trace {total_traces} lacks concept:name')
            seen = set()
            biz = []
            for i, event in enumerate(raw_event_elements):
                attributes = [(str(x.tag), x.get('key'), x.get('value')) for x in event if x.get('key') is not None]
                fields = {key: value for _, key, value in attributes}
                activity = fields.get(ACTIVITY, '')
                timestamp = fields.get(TIME)
                if not activity or not timestamp:
                    continue
                identity = tuple(sorted(attributes))
                if identity in seen:
                    continue
                seen.add(identity)
                raw_events += 1
                if _business_event(activity, fields.get(LIFECYCLE, '')):
                    biz.append((_stamp(timestamp), i, activity, fields.get(RESOURCE)))
                    business_events += 1
            if biz:
                biz.sort(key=lambda e: (e[0], e[1]))
                first_idx = next((j for j, x in enumerate(biz) if x[2] == 'A_Incomplete'), None)
                if first_idx is not None:
                    eligible += 1
                    prefix = biz[:first_idx + 1]
                    future = biz[first_idx + 1:]
                    cutoff = prefix[-1][0]
                    history = [x[2] for x in prefix]
                    cnt = Counter(history)
                    n_future = sum((x[2] == 'W_Validate application' for x in future))
                    resources = [x[3] for x in prefix if x[3] is not None]
                    at_incomplete = prefix[-1][3] or 'Unknown'
                    row = {CASE: case_id, 'Cutoff_Time': cutoff, TARGET: int(n_future >= 2), 'Future_Revalidation_Count': n_future, 'Elapsed_Days_To_First_Incomplete': (cutoff - biz[0][0]).total_seconds() / 86400, 'Prefix_Event_Count': len(prefix), 'Prefix_Unique_Activities': len(cnt), 'Cutoff_Month': cutoff.month, 'Cutoff_DayOfWeek': cutoff.weekday(), 'Cutoff_Hour': cutoff.hour, 'Unique_Resources_So_Far': len(set(resources)), 'Resource_At_Incomplete': at_incomplete, 'Previous_Activity': history[-2] if len(history) >= 2 else 'None'}
                    for col, activity_name in COUNT_FEATURES.items():
                        row[col] = cnt[activity_name]
                    for col in OPTIONAL_CASE_ATTRIBUTES:
                        key = col.split(':', 1)[1]
                        if key in trace_props:
                            val = trace_props[key]
                            row[col] = float(val) if key == 'RequestedAmount' else val
                    rows.append(row)
            trace.clear()
            while trace.getprevious() is not None:
                del trace.getparent()[0]
            if every and total_traces % every == 0:
                print(f'  Parsed {total_traces:,} traces; eligible={eligible:,}', flush=True)
    if not rows:
        raise ValueError('No cases reached A_Incomplete; check event log and lifecycle rules')
    dataset = pd.DataFrame(rows)
    dataset['Cutoff_Time'] = pd.to_datetime(dataset['Cutoff_Time'], utc=True)
    dataset = dataset.sort_values('Cutoff_Time', kind='stable').reset_index(drop=True)
    if 'case:RequestedAmount' in dataset.columns:
        dataset['case:RequestedAmount'] = pd.to_numeric(dataset['case:RequestedAmount'], errors='coerce')
    stats = {'raw_cases': total_traces, 'raw_events_after_dedup': raw_events, 'business_events': business_events, 'eligible_cases': int(len(dataset)), 'positive_cases': int(dataset[TARGET].sum())}
    return (dataset, stats)


def resolve_file_path(filename):
    path = Path(filename).expanduser()
    if path.is_file():
        return path.resolve()
    for folder in (PROJECT_FOLDER, Path.cwd(), PROJECT_FOLDER / "data", Path.cwd() / "data"):
        candidate = folder / filename
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find file: {filename}")


def find_xes():
    if XES_FILE:
        return resolve_file_path(XES_FILE)
    search_folders = (
        PROJECT_FOLDER, Path.cwd(), PROJECT_FOLDER / "data", Path.cwd() / "data"
    )
    found = []
    for folder in search_folders:
        if folder.is_dir():
            for pattern in ("*.xes.gz", "*.xes", "*xes*.gz", "*2017*.gz"):
                found.extend(folder.glob(pattern))
    found = list({p.resolve() for p in found if p.is_file()})
    found.sort(key=lambda p: ("2017" not in p.name, "bpi" not in p.name.lower(), p.name))
    if not found:
        raise FileNotFoundError(
            "BPI Challenge 2017 XES not found. Put the .xes.gz file next to this script "
            "or set XES_FILE at the top of the script."
        )
    return found[0]



# 3. TRAIN THE SIX MODELS

class SklearnCompatibleCatBoost(ClassifierMixin, BaseEstimator):

    def __init__(self, iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=6, auto_class_weights='Balanced', random_seed=42, thread_count=4):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.auto_class_weights = auto_class_weights
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        self.model_ = CatBoostClassifier(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg, auto_class_weights=self.auto_class_weights, random_seed=self.random_seed, thread_count=self.thread_count, loss_function='Logloss', verbose=False, allow_writing_files=False)
        self.model_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return self.model_.predict(X).astype(int).ravel()

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def model_factory(seed: int) -> dict:
    trees = 300
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=trees, max_depth=12, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1
        ),
    }
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError('Please install xgboost to compare all original four models') from exc
    models["XGBoost"] = XGBClassifier(
        n_estimators=trees, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
        eval_metric="logloss", random_state=seed, n_jobs=-1
    )
    models["CatBoost"] = SklearnCompatibleCatBoost(
        iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=6,
        auto_class_weights="Balanced", random_seed=seed, thread_count=4
    )
    return models


def preprocessor_for(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    try:
        one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown='ignore', sparse=False)
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', one_hot)])
    return ColumnTransformer([('num', num_pipe, numeric), ('cat', cat_pipe, categorical)], remainder='drop')


def supervised_fit(model_name, X, y, seed):
    model = model_factory(seed)[model_name]
    if model_name == 'XGBoost':
        pos = int(y.sum())
        neg = len(y) - pos
        model.set_params(scale_pos_weight=neg / max(pos, 1))
    pipeline = Pipeline([('preprocess', preprocessor_for(X)), ('model', model)])
    pipeline.fit(X, y)
    return pipeline


class BanditPolicy(nn.Module):

    def __init__(self, n_inputs: int, hidden: int=48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


class FittedBandit:

    def __init__(self, preprocessing: ColumnTransformer, policy: BanditPolicy):
        self.preprocessing = preprocessing
        self.policy = policy.cpu().eval()

    def risk_score(self, X):
        mat = self.preprocessing.transform(X)
        t = torch.from_numpy(np.asarray(mat, dtype=np.float32))
        with torch.no_grad():
            return torch.softmax(self.policy(t), dim=1)[:, 1].numpy()

    def predict(self, X):
        return (self.risk_score(X) >= 0.5).astype(int)


# 4. EVALUATION METRICS

def classification_metrics(y, scores, *, model_name):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    ypred = (scores >= 0.5).astype(int)
    return {'Model': model_name, 'Accuracy': float(accuracy_score(y, ypred)), 'Precision': float(precision_score(y, ypred, zero_division=0)), 'Recall': float(recall_score(y, ypred, zero_division=0)), 'F1': float(f1_score(y, ypred, zero_division=0)), 'ROC_AUC': float(roc_auc_score(y, scores)), 'Average_Precision(AP)': float(average_precision_score(y, scores))}


# 5. REINFORCE AND RISK-BASED PRIORITIZATION

def train_bandit(X_train, y_train, seed, epochs, X_val=None, y_val=None, hidden=48, lr=0.002, batch_size=256, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    prep = preprocessor_for(X_train)
    Xt = np.asarray(prep.fit_transform(X_train), dtype=np.float32)
    yt = np.asarray(y_train, dtype=np.int64)
    X_tensor = torch.from_numpy(Xt)
    y_tensor = torch.from_numpy(yt)
    pi = BanditPolicy(Xt.shape[1], hidden)
    opt = torch.optim.Adam(pi.parameters(), lr=lr, weight_decay=0.0001)
    pos = float(np.mean(yt))
    class_weights = torch.tensor([0.5 / max(1 - pos, 0.01), 0.5 / max(pos, 0.01)], dtype=torch.float32)
    rng = np.random.default_rng(seed)
    baseline = 0.0
    history = []
    for ep in range(1, epochs + 1):
        pi.train()
        index = rng.permutation(len(yt))
        rewards_this_epoch = []
        for start in range(0, len(index), batch_size):
            idx = index[start:start + batch_size]
            Xb = X_tensor[idx]
            yb = y_tensor[idx]
            logits = pi(Xb)
            dist = torch.distributions.Categorical(logits=logits)
            chosen = dist.sample()
            reward = ((chosen == yb).float() * 2 - 1) * class_weights[yb]
            baseline = 0.9 * baseline + 0.1 * float(reward.detach().mean())
            advantage = reward.detach() - baseline
            loss = -(dist.log_prob(chosen) * advantage).mean() - 0.012 * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pi.parameters(), 5.0)
            opt.step()
            rewards_this_epoch.append(float(reward.detach().mean()))
        entry = {'Epoch': ep, 'Mean_Weighted_Reward': np.mean(rewards_this_epoch)}
        if X_val is not None:
            pi.eval()
            valmat = prep.transform(X_val)
            with torch.no_grad():
                probs = torch.softmax(pi(torch.from_numpy(np.asarray(valmat, dtype=np.float32))), dim=1)[:, 1].numpy()
            entry['Validation_F1'] = float(f1_score(y_val, (probs >= 0.5).astype(int), zero_division=0))
        history.append(entry)
        if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
            report = f"  Bandit epoch {ep:3d}/{epochs} reward={entry['Mean_Weighted_Reward']:+.3f}"
            if X_val is not None:
                report += f" val F1={entry['Validation_F1']:.3f}"
            print(report, flush=True)
    return (FittedBandit(prep, pi), pd.DataFrame(history))


def rank_and_baselines(y, scores, rule_feature, case_ids, cutoff_times, capacities=CAPACITIES, random_reps=1000, seed=SEED):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    rule = np.asarray(rule_feature, dtype=float)
    n = len(y)
    selection = pd.DataFrame({
        '_Risk': scores, '_Rule': rule,
        '_Case': pd.Series(case_ids).astype(str).to_numpy(),
        '_Time': pd.to_datetime(cutoff_times, utc=True),
    })
    ml_order = selection.sort_values(
        ['_Risk', '_Time', '_Case'], ascending=[False, True, True], kind='mergesort'
    ).index.to_numpy()
    rule_order = selection.sort_values(
        ['_Rule', '_Time', '_Case'], ascending=[False, True, True], kind='mergesort'
    ).index.to_numpy()
    records = []
    chosen = {}
    for frac in capacities:
        k = int(math.ceil(frac * n))
        cap = int(round(100 * frac))
        for name, order in [('Selected model', ml_order), ('Rule-Based (Prefix Event Count)', rule_order)]:
            selected = order[:k]
            caught = int(y[selected].sum())
            records.append({'Capacity_Pct': cap, 'Strategy': name, 'K': k, 'Captured_Rework': caught, 'Precision_at_K': caught / k, 'Recall_at_K': caught / max(1, int(y.sum())), 'Lift_at_K': caught / k / max(y.mean(), 1e-12), 'False_Alerts': k - caught})
            if name == 'Selected model':
                chosen[cap] = selected
        rng = np.random.default_rng(seed + k)
        random_caps = np.empty(random_reps, dtype=np.float64)
        for i in range(random_reps):
            selection = rng.choice(n, size=k, replace=False)
            random_caps[i] = y[selection].sum()
        caught = float(random_caps.mean())
        records.append({'Capacity_Pct': cap, 'Strategy': f'Random ({random_reps} draws)', 'K': k, 'Captured_Rework': caught, 'Precision_at_K': caught / k, 'Recall_at_K': caught / max(1, int(y.sum())), 'Lift_at_K': caught / k / max(y.mean(), 1e-12), 'False_Alerts': k - caught})
    return (pd.DataFrame(records), chosen)


def plot_winner_vs_baselines(df, winner, outdir):
    names = df['Strategy'].tolist()
    values = df['Captured_Rework'].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(names)), values)
    ax.set_xticks(range(len(names)), names, rotation=12, ha='right')
    ax.set_ylabel('Actual rework cases captured')
    ax.set_title(f'{winner} vs Baselines — 10% Review Capacity')
    fig.tight_layout()
    fig.savefig(outdir / 'Winner_Risk_Prioritization_Comparison.png', dpi=190)
    plt.close(fig)


def plot_classification_comparison(df, outdir):
    models = df["Model"].tolist()
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 0.2, df["F1"], width=0.4, label="F1-score")
    ax.bar(x + 0.2, df["ROC_AUC"], width=0.4, label="ROC-AUC")
    ax.set_xticks(x, models, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Six-model comparison on the chronological test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "Six_Model_Classification_Comparison.png", dpi=190)
    plt.close(fig)


# 6. TRAIN, COMPARE, AND PRIORITIZE

def run(args):
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Build a fresh predictive dataset from the original event log
    path = find_xes()
    print('Reading real XES:', path, flush=True)
    dataset, stats = parse_xes_and_build_dataset(path)
    dataset.to_csv(output_dir / 'Task4_Predictive_Dataset.csv', index=False)
    print('Dataset:', stats, flush=True)
    needed = [CASE, 'Cutoff_Time', TARGET, 'Future_Revalidation_Count', 'Prefix_Event_Count']
    missing = [c for c in needed if c not in dataset.columns]
    if missing:
        raise ValueError(f'Predictive dataset missing columns: {missing}')
    if dataset[CASE].isna().any() or dataset[CASE].duplicated().any():
        raise ValueError('Expected exactly one unique row per case')
    if not dataset[TARGET].isin([0, 1]).all():
        raise ValueError('Target must be binary')
    # Chronological 70/30 split; validation is within the training period
    train_end = int(0.7 * len(dataset))
    dev = dataset.iloc[:train_end].copy()
    test = dataset.iloc[train_end:].copy()
    inner_train_end = int(len(dev) * 0.8)
    inner = dev.iloc[:inner_train_end].copy()
    validation = dev.iloc[inner_train_end:].copy()
    drop_cols = [CASE, 'Cutoff_Time', TARGET, 'Future_Revalidation_Count']
    features = [c for c in dataset.columns if c not in drop_cols]
    assert TARGET not in features and 'Future_Revalidation_Count' not in features
    X_inner = inner[features]
    y_inner = inner[TARGET].to_numpy(dtype=int)
    X_val = validation[features]
    y_val = validation[TARGET].to_numpy(dtype=int)
    X_dev = dev[features]
    y_dev = dev[TARGET].to_numpy(dtype=int)
    X_test = test[features]
    y_test = test[TARGET].to_numpy(dtype=int)
    if any((len(np.unique(y)) != 2 for y in (y_inner, y_val, y_dev, y_test))):
        raise ValueError('Chronological split lacks a class in one partition.')
    print(f'Chronological split: inner train={len(inner):,} | validation={len(validation):,} | test={len(test):,}', flush=True)
    # Compare the six models on validation and select the highest F1
    print('Training six models and selecting best Validation F1.', flush=True)
    model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'CatBoost']
    validation_rows = []
    for name in model_names:
        print('Validation fit:', name, flush=True)
        model = supervised_fit(name, X_inner, y_inner, seed)
        scores = model.predict_proba(X_val)[:, 1]
        validation_rows.append(classification_metrics(y_val, scores, model_name=name))
    print('Validation fit: Contextual Bandit (REINFORCE)', flush=True)
    bandit, bandit_curve = train_bandit(
        X_inner, y_inner, seed, epochs=args.bandit_epochs,
        X_val=X_val, y_val=y_val, hidden=args.bandit_hidden, verbose=True,
    )
    best_epoch = int(bandit_curve.sort_values(
        ['Validation_F1', 'Epoch'], ascending=[False, True]
    ).iloc[0]['Epoch'])
    if best_epoch < len(bandit_curve):
        bandit, _ = train_bandit(
            X_inner, y_inner, seed, epochs=best_epoch,
            hidden=args.bandit_hidden, verbose=False,
        )
    val_scores = bandit.risk_score(X_val)
    validation_rows.append(classification_metrics(
        y_val, val_scores, model_name='Contextual Bandit (RL-family)'
    ))
    validation_df = pd.DataFrame(validation_rows).sort_values(
        ['F1', 'ROC_AUC', 'Model'], ascending=[False, False, True]
    ).reset_index(drop=True)
    validation_df.to_csv(output_dir / 'Validation_Model_Comparison.csv', index=False)
    bandit_curve.to_csv(output_dir / 'Bandit_Validation_Learning_Curve.csv', index=False)
    winner = validation_df.iloc[0]['Model']
    print('\nBEST MODEL SELECTED ON VALIDATION F1:', winner, flush=True)
    print('Bandit best epoch on validation:', best_epoch, flush=True)
    print(validation_df.round(4).to_string(index=False), flush=True)
    # Refit all six models and evaluate on the held-out test set
    model_test_scores = {}
    final_supervised = {}
    for name in model_names:
        print('Final fit on full 70%:', name, flush=True)
        fitted = supervised_fit(name, X_dev, y_dev, seed)
        final_supervised[name] = fitted
        model_test_scores[name] = fitted.predict_proba(X_test)[:, 1]
        joblib.dump(fitted, output_dir / f"Model_{name.replace(' ', '_')}.joblib")
    print('Final fit on full 70%: Contextual Bandit', flush=True)
    final_bandit, _ = train_bandit(X_dev, y_dev, seed, epochs=best_epoch, hidden=args.bandit_hidden, verbose=True)
    model_test_scores['Contextual Bandit (RL-family)'] = final_bandit.risk_score(X_test)
    joblib.dump(final_bandit.preprocessing, output_dir / 'Bandit_Preprocessing.joblib')
    torch.save({'state_dict': final_bandit.policy.state_dict(), 'input_features': int(final_bandit.policy.net[0].in_features), 'hidden': args.bandit_hidden, 'best_epochs': best_epoch}, output_dir / 'Bandit_Policy_State.pt')
    # Compare classification performance for all six models on test data
    test_metrics = []
    for name, scores in model_test_scores.items():
        test_metrics.append(classification_metrics(y_test, scores, model_name=name))
        matrix = confusion_matrix(y_test, (scores >= 0.5).astype(int), labels=[0, 1])
        pd.DataFrame(
            matrix,
            index=['Actual_No_Rework', 'Actual_Rework'],
            columns=['Pred_No_Rework', 'Pred_Rework'],
        ).to_csv(output_dir / f"Confusion_Matrix_{name.replace(' ', '_').replace('/', '_')}.csv")
    pd.DataFrame(test_metrics).to_csv(output_dir / 'Test_Model_Comparison_All_6.csv', index=False)
    plot_classification_comparison(pd.DataFrame(test_metrics), output_dir)
    case_predictions = test[[CASE, 'Cutoff_Time', TARGET]].copy().reset_index(drop=True)
    for name, scores in model_test_scores.items():
        slug = name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        case_predictions[f'Prediction_Score_{slug}'] = scores
    case_predictions.to_csv(output_dir / 'Test_Case_Predictions_All_6.csv', index=False)

    # Risk-based prioritization: ONLY the validation-selected best model
    winner_scores = model_test_scores[winner]
    ranked = test[[CASE, 'Cutoff_Time', TARGET, 'Prefix_Event_Count']].copy().reset_index(drop=True)
    ranked['Risk_Score'] = winner_scores
    ranked['Risk_Score_Description'] = (
        'Action_Policy_Probability_Not_Calibrated' if winner.startswith('Contextual Bandit')
        else 'Predicted_Rework_Probability_Not_Externally_Calibrated'
    )
    ranked['_Case_Tie_Key'] = ranked[CASE].astype(str)
    ranked = ranked.sort_values(
        ['Risk_Score', 'Cutoff_Time', '_Case_Tie_Key'],
        ascending=[False, True, True], kind='mergesort'
    ).reset_index(drop=True)
    ranked['Risk_Rank'] = np.arange(1, len(ranked) + 1)
    selected_count = int(math.ceil(REVIEW_CAPACITY * len(ranked)))
    ranked['Selected_Top_10pct'] = ranked['Risk_Rank'] <= selected_count
    ranked.drop(columns='_Case_Tie_Key').to_csv(
        output_dir / 'Winner_Test_Case_Ranking.csv', index=False
    )
    ranked.loc[ranked['Selected_Top_10pct']].drop(columns='_Case_Tie_Key').to_csv(
        output_dir / 'Winner_Top_10pct_Cases.csv', index=False
    )
    # All baseline methods receive the same fixed review budget
    baseline_df, _ = rank_and_baselines(
        y_test, winner_scores, test['Prefix_Event_Count'].to_numpy(),
        test[CASE].to_numpy(), test['Cutoff_Time'].to_numpy(),
        random_reps=args.random_repetitions, seed=seed,
    )
    baseline_df['Strategy'] = baseline_df['Strategy'].replace(
        {'Selected model': winner}
    )
    baseline_df.to_csv(output_dir / 'Winner_Risk_Prioritization_vs_Baselines.csv', index=False)
    plot_winner_vs_baselines(baseline_df, winner, output_dir)
    if args.feature_importance:
        print('Computing descriptive LR permutation importance...', flush=True)
        imp = permutation_importance(final_supervised['Logistic Regression'], X_test, y_test, scoring='f1', n_repeats=5, random_state=seed, n_jobs=-1)
        dfimp = pd.DataFrame({'Feature': features, 'Importance_Mean': imp.importances_mean, 'Importance_STD': imp.importances_std})
        dfimp.sort_values('Importance_Mean', ascending=False).to_csv(output_dir / 'Logistic_Regression_Feature_Importance.csv', index=False)
    # Save reproducible experiment settings and summary tables
    config = {
        "dataset_stats": stats,
        "split": {
            "inner_training": len(inner), "validation": len(validation),
            "final_development": len(dev), "test": len(test),
            "type": "chronological 70/30; validation last 20% of development",
        },
        "validation_selection_metric": "F1-score",
        "winner_chosen_on_validation": winner,
        "bandit_best_epochs": best_epoch,
        "bandit_not_sequential_rl": True,
        "risk_scores_not_necessarily_calibrated": True,
        "intervention_effectiveness_not_tested": True,
        "features": features,
        "review_capacity_pct": 10,
        "seed": seed,
        "random_repetitions": args.random_repetitions,
        "models": [
            "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost",
            "CatBoost", "Contextual Bandit (REINFORCE)",
        ],
    }
    (output_dir / 'Experiment_Config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    report = [
        "# CPIS 603 — Six-model prediction and prioritization experiment",
        "",
        "**Dataset:** REAL BPI Challenge 2017",
        f"**Validation-selected winner:** {winner}",
        "**Model selection:** highest Validation F1; test outcomes not used.",
        f"**Split:** inner train {len(inner)}, validation {len(validation)}, "
        f"final development {len(dev)}, test {len(test)}.",
        f"**Bandit selected training epochs:** {best_epoch}",
        "**Prediction target:** at least two future W_Validate application "
        "events after first A_Incomplete.",
        "",
        "## Validation comparison",
        validation_df.round(4).to_markdown(index=False),
        "",
        "## Test comparison",
        pd.DataFrame(test_metrics).round(4).to_markdown(index=False),
        "",
        "## Selected model ONLY versus baselines at 10%",
        baseline_df.round(4).to_markdown(index=False),
        "",
    ]
    (output_dir / 'Paper_Ready_Experiment_Report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n==== FINAL RESULTS (do not choose model after seeing test) ====', flush=True)
    print(pd.DataFrame(test_metrics).round(4).to_string(index=False), flush=True)
    print('\n==== BEST MODEL ONLY VS RULE-BASED & RANDOM (10%) ==== ', flush=True)
    print(baseline_df.round(4).to_string(index=False), flush=True)
    print('Results saved:', output_dir, flush=True)
    return output_dir




if __name__ == "__main__":
    settings = SimpleNamespace(
        output=str(PROJECT_FOLDER / OUTPUT_FOLDER),
        seed=RANDOM_SEED,
        bandit_epochs=BANDIT_MAX_EPOCHS,
        bandit_hidden=BANDIT_HIDDEN_UNITS,
        random_repetitions=RANDOM_REPETITIONS,
        feature_importance=SAVE_FEATURE_IMPORTANCE,
    )
    run(settings)


