import os, sys, time, math, random, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
)
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------- КОНФИГУРАЦИЯ -------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

AGENT_TYPES = ["ground", "aerial", "amphibious"]
POWER_RANGE = (0.5, 2.5)
COVERAGE_RANGE = (0.2, 2.0)
MOBILITY_RANGE = (0.0, 1.5)
COST_RANGE = (1.0, 5.0)

MC_GT_ITER = 1000          # для тестового прогона; финал 10000
MC_FAST_ITER = 10          # финал 50
N_FAST_FEATURES = 5        # финал 20

N_SPLITS = 5               # финал 20
TEST_SIZE = 0.2

XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'verbosity': 0
}

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings('ignore')

# ------------------------- ГЕНЕРАЦИЯ АГЕНТОВ -------------------------
def generate_agents(n, type_distribution=None):
    agents = []
    if type_distribution is not None:
        assert len(type_distribution) == len(AGENT_TYPES)
        assert sum(type_distribution) == n
        type_list = []
        for t, count in zip(AGENT_TYPES, type_distribution):
            type_list.extend([t] * count)
        random.shuffle(type_list)
        for idx, agent_type in enumerate(type_list):
            agents.append(_make_agent(idx, agent_type))
    else:
        for i in range(n):
            agent_type = random.choice(AGENT_TYPES)
            agents.append(_make_agent(i, agent_type))
    return agents

def _make_agent(agent_id, agent_type):
    power = round(random.uniform(*POWER_RANGE), 3)
    coverage = round(random.uniform(*COVERAGE_RANGE), 3)
    mobility = round(random.uniform(*MOBILITY_RANGE), 3)
    cost = round(random.uniform(*COST_RANGE), 2)
    return {
        'id': agent_id,
        'type': agent_type,
        'features': {'power': power, 'coverage': coverage, 'mobility': mobility},
        'cost': cost
    }

def feat_vec(agent):
    f = agent['features']
    return np.array([f['power'], f['coverage'], f['mobility']])

# ------------------------- ХАРАКТЕРИСТИЧЕСКИЕ ФУНКЦИИ -------------------------
def smooth_synergy(coalition, weight_vector=None, synergy_lambda=0.5, **kwargs):
    if not coalition: return 0.0
    if weight_vector is None: weight_vector = np.array([1.0, 1.0, 1.0])
    value = sum(np.dot(weight_vector, feat_vec(a)) for a in coalition)
    synergy = 0.0
    for a1, a2 in combinations(coalition, 2):
        x1 = feat_vec(a1); x2 = feat_vec(a2)
        cos_sim = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-8)
        synergy += synergy_lambda * cos_sim
    return value + synergy

def search_rescue(coalition, area_size=20.0, time_limit=10.0, **kwargs):
    if not coalition: return 0.0
    total_coverage = sum(a['features']['coverage'] for a in coalition)
    eff_area = area_size * (1 - np.exp(-total_coverage / area_size))
    total_power = sum(a['features']['power'] for a in coalition)
    sensor_factor = 0.1 * total_power if total_power < 5 else 1.0 + 0.5 * np.log(1 + total_power - 5)
    total_mobility = sum(a['features']['mobility'] for a in coalition)
    search_time = min(time_limit, total_mobility * 0.5)
    time_factor = search_time / time_limit
    types = {a['type'] for a in coalition}
    type_bonus = {1: 1.0, 2: 1.3, 3: 1.6}.get(len(types), 1.0)
    base_prob = eff_area / area_size
    adjusted = base_prob * sensor_factor * time_factor * type_bonus
    return min(0.99, adjusted)

def threshold_voting(coalition, threshold=50.0, **kwargs):
    if not coalition: return 0.0
    total_power = sum(a['features']['power'] for a in coalition)
    return 1.0 if total_power >= threshold else 0.0

def complementary_game(coalition, threshold=10.8, **kwargs):
    if len(coalition) < 3: return 0.0
    types = {a['type'] for a in coalition}
    total_power = sum(a['features']['power'] for a in coalition)
    if types == set(AGENT_TYPES) and total_power >= threshold:
        return 10.0
    return 0.0

# ------------------------- ВЫЧИСЛЕНИЕ ШЕПЛИ -------------------------
def exact_shapley(agent_id, agents, coalition_func, **kwargs):
    n = len(agents)
    other_ids = [a['id'] for a in agents if a['id'] != agent_id]
    cache = {}
    total = 0.0
    for r in range(len(other_ids)+1):
        for subset_ids in combinations(other_ids, r):
            S = frozenset(subset_ids)
            if S not in cache:
                cache[S] = coalition_func([a for a in agents if a['id'] in S], **kwargs)
            S_with = frozenset(subset_ids + (agent_id,))
            if S_with not in cache:
                cache[S_with] = coalition_func([a for a in agents if a['id'] in S_with], **kwargs)
            marginal = cache[S_with] - cache[S]
            weight = (math.factorial(len(subset_ids)) *
                      math.factorial(n - len(subset_ids) - 1) / math.factorial(n))
            total += weight * marginal
    return total

def mc_shapley(agent_id, agents, coalition_func, m, rng, **kwargs):
    ids = [a['id'] for a in agents]
    id_to_agent = {a['id']: a for a in agents}
    total = 0.0
    for _ in range(m):
        perm = rng.permutation(ids)
        pos = np.where(perm == agent_id)[0][0]
        Sb = [id_to_agent[i] for i in perm[:pos]]
        Sa = Sb + [id_to_agent[agent_id]]
        total += coalition_func(Sa, **kwargs) - coalition_func(Sb, **kwargs)
    return total / m

# ------------------------- ПРИЗНАКИ -------------------------
def compute_features(agents, coalition_func, fast_mc=False,
                     fast_mc_iter=None, n_fast_features=None, rng=None, **kwargs):
    n = len(agents)
    if fast_mc_iter is None: fast_mc_iter = MC_FAST_ITER
    if n_fast_features is None: n_fast_features = N_FAST_FEATURES
    if rng is None: rng = np.random.RandomState(RANDOM_SEED)

    all_fv = np.array([feat_vec(a) for a in agents])
    mean_fv = np.mean(all_fv, axis=0)
    std_fv = np.std(all_fv, axis=0)
    type_counts = {t: sum(1 for a in agents if a['type'] == t) for t in AGENT_TYPES}

    X = np.zeros((n, 16))
    for i, a in enumerate(agents):
        fv = feat_vec(a)
        X[i, 0] = fv[0]
        X[i, 1] = fv[1]
        X[i, 2] = fv[2]
        X[i, 3] = a['cost']
        X[i, 4] = 1.0 if a['type'] == 'ground' else 0.0
        X[i, 5] = 1.0 if a['type'] == 'aerial' else 0.0
        X[i, 6] = 1.0 if a['type'] == 'amphibious' else 0.0
        X[i, 7] = mean_fv[0]
        X[i, 8] = mean_fv[1]
        X[i, 9] = mean_fv[2]
        X[i,10] = std_fv[0]
        X[i,11] = std_fv[1]
        X[i,12] = std_fv[2]
        X[i,13] = type_counts['ground']
        X[i,14] = type_counts['aerial']
        X[i,15] = type_counts['amphibious']

    # Всегда используем Монте-Карло; для малых n берём больше итераций
    mc_iter = MC_GT_ITER if n > 20 else 50000
    y = np.zeros(n)
    for i, a in enumerate(agents):
        y[i] = mc_shapley(a['id'], agents, coalition_func, mc_iter, rng, **kwargs)

    if fast_mc:
        fast_feat = np.zeros((n, n_fast_features))
        for i, a in enumerate(agents):
            for j in range(n_fast_features):
                fast_feat[i, j] = mc_shapley(a['id'], agents, coalition_func,
                                             fast_mc_iter, rng, **kwargs)
        X = np.hstack([X, fast_feat])
    return X, y

# ------------------------- МОДЕЛИ -------------------------
def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge(X_train, y_train):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model

def train_dummy(X_train, y_train):
    model = DummyRegressor(strategy='mean')
    model.fit(X_train, y_train)
    return model

def train_kmeans(X_train, y_train):
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=10).fit(X_train)
    cluster_means = {cl: np.mean(y_train[kmeans.labels_ == cl]) for cl in np.unique(kmeans.labels_)}
    class KMeansPredictor:
        def predict(self, X): return np.array([cluster_means[l] for l in kmeans.predict(X)])
    return KMeansPredictor()

# ------------------------- МЕТРИКИ -------------------------
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = 100.0 * np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-10))
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MedAE': medae, 'SMAPE': smape}

# ------------------------- КРОСС-ВАЛИДАЦИЯ -------------------------
def evaluate_models(X, y, models, names, n_splits=N_SPLITS, test_size=TEST_SIZE, rng=None):
    if rng is None: rng = np.random.RandomState(RANDOM_SEED)
    results = {name: [] for name in names}
    for split in range(n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=rng.randint(0,99999))
        for model_func, name in zip(models, names):
            model = model_func(X_tr, y_tr)
            pred = model.predict(X_te)
            met = compute_metrics(y_te, pred)
            results[name].append(met)
    summary = {}
    for name in names:
        summary[name] = {}
        for key in results[name][0]:
            vals = [r[key] for r in results[name]]
            summary[name][key] = (np.mean(vals), np.std(vals))
    return summary

# ------------------------- ГРАФИКИ -------------------------
def plot_true_vs_pred(y_true, y_pred, title, filename):
    plt.figure(figsize=(6,5))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k', s=30)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Истинное значение Шепли')
    plt.ylabel('Предсказанное значение')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_error_boxplot(errors_dict, title, filename):
    df = pd.DataFrame(errors_dict)
    plt.figure(figsize=(8,5))
    df.boxplot()
    plt.ylabel('Абсолютная ошибка')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_xgb_learning_curve(model, X_tr, y_tr, X_val, y_val, filename):
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=False)
    results = model.evals_result()
    if not results:
        return
    plt.figure(figsize=(8,5))
    for key, values in results['validation_0'].items():
        plt.plot(values, label=key)
    plt.xlabel('Число деревьев')
    plt.ylabel('Ошибка')
    plt.title('Кривая обучения XGBoost')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_linearity_spectrum(metrics_dict, filename):
    games = list(metrics_dict.keys())
    lr_vals = [metrics_dict[g]['LinReg R2'] for g in games]
    xgb_vals = [metrics_dict[g]['XGBoost R2'] for g in games]
    x = np.arange(len(games))
    width = 0.35
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, lr_vals, width, label='Линейная регрессия')
    plt.bar(x + width/2, xgb_vals, width, label='XGBoost')
    plt.ylabel('R²')
    plt.title('Спектр линейности')
    plt.xticks(x, games, rotation=45)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# ------------------------- ЭТАПЫ -------------------------
def stage0_verification():
    print("\n=== ЭТАП 0: ВЕРИФИКАЦИЯ ===")
    n = 5
    for func, fname, kwargs in [
        (smooth_synergy, "smooth_synergy", {}),
        (search_rescue, "search_rescue", {}),
        (threshold_voting, "threshold_voting", {'threshold': 20.0}),
        (complementary_game, "complementary_game", {'threshold': 5.0})
    ]:
        agents = generate_agents(n)
        exact = np.array([exact_shapley(a['id'], agents, func, **kwargs) for a in agents])
        rng = np.random.RandomState(RANDOM_SEED)
        mc = np.array([mc_shapley(a['id'], agents, func, MC_GT_ITER, rng, **kwargs) for a in agents])
        max_err = np.max(np.abs(exact - mc))
        mean_err = np.mean(np.abs(exact - mc))
        print(f"{fname}: max_err={max_err:.6f}, mean_err={mean_err:.6f}")

def stage1_baseline():
    print("\n=== ЭТАП 1: БАЗОВОЕ СРАВНЕНИЕ (20 агентов, гладкая синергия) ===")
    agents = generate_agents(20)
    X, y = compute_features(agents, smooth_synergy)
    models = [train_dummy, train_linear, train_ridge, train_xgboost, train_kmeans]
    names = ["Dummy", "Linear Regression", "Ridge", "XGBoost", "KMeans"]
    results = evaluate_models(X, y, models, names)
    df = pd.DataFrame({name: {k: f"{v[0]:.4f}±{v[1]:.4f}" for k,v in m.items()} for name,m in results.items()}).T
    df.to_csv(os.path.join(OUTPUT_DIR, "stage1.csv"))
    print(df)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = train_linear(X_tr, y_tr)
    lr_pred = lr_model.predict(X_te)
    plot_true_vs_pred(y_te, lr_pred, "Линейная регрессия (гл. синергия)", "stage1_scatter_linreg.png")

    xgb_model = train_xgboost(X_tr, y_tr)
    xgb_pred = xgb_model.predict(X_te)
    plot_true_vs_pred(y_te, xgb_pred, "XGBoost (гл. синергия)", "stage1_scatter_xgb.png")

    errors = {
        'Linear': np.abs(y_te - lr_pred),
        'XGBoost': np.abs(y_te - xgb_pred),
    }
    plot_error_boxplot(errors, "Ошибки на тестовой выборке (гл. синергия)", "stage1_error_boxplot.png")
    plot_xgb_learning_curve(xgb_model, X_tr, y_tr, X_te, y_te, "stage1_xgb_learning.png")

def stage2_search_rescue():
    print("\n=== ЭТАП 2: ПОИСК И СПАСЕНИЕ (100 агентов) ===")
    agents = generate_agents(100)
    X_base, y = compute_features(agents, search_rescue)
    X_full, _ = compute_features(agents, search_rescue, fast_mc=True)
    models = [train_linear, train_ridge, train_xgboost]
    names = ["Linear Regression", "Ridge", "XGBoost Hybrid"]
    results = evaluate_models(X_full, y, models, names)
    rng = np.random.RandomState(RANDOM_SEED)
    mc_preds = np.array([mc_shapley(a['id'], agents, search_rescue, 200, rng) for a in agents])
    mc_metrics = compute_metrics(y, mc_preds)
    results["MC 200 iter"] = {k: (v, 0.0) for k,v in mc_metrics.items()}
    df = pd.DataFrame({name: {k: f"{v[0]:.4f}±{v[1]:.4f}" for k,v in m.items()} for name,m in results.items()}).T
    df.to_csv(os.path.join(OUTPUT_DIR, "stage2.csv"))
    print(df)

    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.2, random_state=42)
    lr_model = train_linear(X_tr, y_tr)
    lr_pred = lr_model.predict(X_te)
    plot_true_vs_pred(y_te, lr_pred, "Линейная регрессия (поиск и спасение)", "stage2_scatter_linreg.png")

    xgb_model = train_xgboost(X_tr, y_tr)
    xgb_pred = xgb_model.predict(X_te)
    plot_true_vs_pred(y_te, xgb_pred, "XGBoost (поиск и спасение)", "stage2_scatter_xgb.png")

    errors = {
        'Linear': np.abs(y_te - lr_pred),
        'XGBoost': np.abs(y_te - xgb_pred),
        'MC 200': np.abs(y_te - mc_preds[X_te.shape[0]//2])  # упрощение
    }
    plot_error_boxplot(errors, "Ошибки (поиск и спасение)", "stage2_error_boxplot.png")
    plot_xgb_learning_curve(xgb_model, X_tr, y_tr, X_te, y_te, "stage2_xgb_learning.png")

def stage3_size_effect():
    print("\n=== ЭТАП 3: ВЛИЯНИЕ РАЗМЕРА КОАЛИЦИИ (все игры) ===")
    functions = [
        ("Smooth synergy", smooth_synergy, {}),
        ("Search & rescue", search_rescue, {}),
        ("Threshold voting", threshold_voting, {"frac": 0.5}),
        ("Complementary", complementary_game, {"frac": 0.5})
    ]
    sizes = [10, 20, 50, 100]
    all_results = []
    for fname, func, params in functions:
        for n in sizes:
            agents = generate_agents(n)
            kwargs = {}
            if "frac" in params:
                max_pow = sum(a['features']['power'] for a in agents)
                kwargs['threshold'] = params["frac"] * max_pow
            X_full, y = compute_features(agents, func, fast_mc=True, **kwargs)
            lr_r2 = []
            xgb_r2 = []
            for _ in range(3):
                X_tr, X_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.2,
                                                          random_state=random.randint(0,99999))
                lr = train_linear(X_tr, y_tr)
                lr_r2.append(r2_score(y_te, lr.predict(X_te)))
                xgb_m = train_xgboost(X_tr, y_tr)
                xgb_r2.append(r2_score(y_te, xgb_m.predict(X_te)))
            all_results.append({
                'Function': fname, 'n': n,
                'LinReg R2': np.mean(lr_r2), 'XGBoost R2': np.mean(xgb_r2)
            })
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUTPUT_DIR, "stage3.csv"), index=False)
    print(df)
    avg_by_game = df.groupby('Function')[['LinReg R2', 'XGBoost R2']].mean()
    plot_linearity_spectrum(avg_by_game.to_dict('index'), "linearity_spectrum.png")

# ------------------------- MAIN -------------------------
if __name__ == "__main__":
    start = time.time()
    stage0_verification()
    stage1_baseline()
    stage2_search_rescue()
    stage3_size_effect()
    print(f"\nВсе этапы завершены за {time.time()-start:.1f} сек.")
    print("Графики и таблицы сохранены в", OUTPUT_DIR)
