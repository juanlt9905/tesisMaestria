"""
comparacion_fs.py

Comparación de estrategias de selección de características.
Todos los selectores comparten la firma:
    selector(X, y, nombres, k, **kwargs) -> list[str]
para que el arnés de evaluación sea agnóstico al método.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import lightgbm as lgb
from scipy.stats import rankdata

import funciones  # módulo existente del pipeline del AG


# ============================================================
# 1. RANKINGS UNIVARIADOS
# ============================================================

def ranking_correlacion(X, y, metodo="spearman"):
    """
    Calcula |correlación| de cada columna de X con el objetivo binario y.

    Implementación vectorizada: para Spearman se rangotransforma X y luego
    se aplica Pearson columna a columna. Evita construir la matriz 190x190.

    Nota: con y binaria, el ranking por |Spearman| es equivalente al ranking
    por |AUC univariada - 0.5| (relación de Mann-Whitney). Útil para
    interpretar el filtro como 'poder discriminante individual'.

    Args:
        X: array (n_muestras, n_variables)
        y: array (n_muestras,) binario
        metodo: 'spearman' (rangos) o 'pearson' (valores crudos)

    Returns:
        array (n_variables,) con los valores absolutos de correlación
    """
    Xr = np.apply_along_axis(rankdata, 0, X) if metodo == "spearman" else X.astype(np.float64)

    Xc = Xr - Xr.mean(axis=0)
    yc = y.astype(np.float64) - y.mean()

    numerador = Xc.T @ yc
    denominador = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denominador > 0, numerador / denominador, 0.0)

    return np.abs(corr)


def ranking_mutual_info(X, y, muestra=200_000, semilla=42):
    """
    Información mutua entre cada variable y el objetivo.
    Submuestrea porque mutual_info_classif es O(n log n) por variable
    y con 750k x 190 es prohibitivo.
    """
    rng = np.random.default_rng(semilla)
    if X.shape[0] > muestra:
        idx = rng.choice(X.shape[0], size=muestra, replace=False)
        X, y = X[idx], y[idx]
    return mutual_info_classif(X, y, random_state=semilla)


# ============================================================
# 2. SELECTORES
# ============================================================

def _top_k(puntajes, nombres, k):
    """Devuelve los k nombres con mayor puntaje, en orden descendente."""
    orden = np.argsort(puntajes)[::-1][:k]
    return [nombres[i] for i in orden]


def selector_correlacion(X, y, nombres, k, metodo="spearman", puntajes=None):
    """Top-k por |correlación| con el objetivo. Filtro univariado ingenuo."""
    if puntajes is None:
        puntajes = ranking_correlacion(X, y, metodo=metodo)
    return _top_k(puntajes, nombres, k)


def selector_mutual_info(X, y, nombres, k, puntajes=None, **kwargs):
    """Top-k por información mutua. Captura dependencias no monótonas."""
    if puntajes is None:
        puntajes = ranking_mutual_info(X, y, **kwargs)
    return _top_k(puntajes, nombres, k)


def selector_mrmr_simple(X, y, nombres, k, umbral_redundancia=0.8,
                         puntajes=None, muestra=100_000, semilla=42):
    """
    Filtro con control de redundancia: recorre las variables en orden de
    relevancia y descarta las que superan `umbral_redundancia` de |rho| con
    alguna variable ya seleccionada.

    Es el baseline honesto frente al AG: ambos consideran interacciones
    entre variables, aunque este lo hace de forma voraz y por pares.
    """
    if puntajes is None:
        puntajes = ranking_correlacion(X, y)

    rng = np.random.default_rng(semilla)
    idx_muestra = (rng.choice(X.shape[0], size=muestra, replace=False)
                   if X.shape[0] > muestra else slice(None))
    Xr = np.apply_along_axis(rankdata, 0, X[idx_muestra])

    orden = np.argsort(puntajes)[::-1]
    seleccionados = []

    for j in orden:
        if len(seleccionados) >= k:
            break
        if not seleccionados:
            seleccionados.append(j)
            continue
        rho = np.corrcoef(Xr[:, j], Xr[:, seleccionados], rowvar=False)[0, 1:]
        if np.max(np.abs(rho)) < umbral_redundancia:
            seleccionados.append(j)

    return [nombres[j] for j in seleccionados]


def selector_aleatorio(X, y, nombres, k, semilla=42):
    """Control nulo: k variables al azar. Promediar sobre varias semillas."""
    rng = np.random.default_rng(semilla)
    return [nombres[i] for i in rng.choice(len(nombres), size=k, replace=False)]


def selector_ag(ruta_resumen, **kwargs):
    """
    Envuelve la extracción del AG. Ignora k: el número de variables lo
    determina la solución de compromiso (punto utópico).
    """
    return funciones.extraccion_caracteristicas_compromiso(ruta_resumen)


# ============================================================
# 3. ARNÉS DE EVALUACIÓN
# ============================================================

PARAMS_LGBM = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)


def evaluar_subconjunto(vars_sel, X_train, y_train, X_val, y_val,
                        nombres, params=None):
    """
    Entrena un LGBM sobre el subconjunto y devuelve métricas de validación.

    El modelo y sus hiperparámetros son IDÉNTICOS para todos los selectores.
    Cualquier diferencia observada es atribuible al subconjunto, no al modelo.

    Returns:
        dict con n_variables, roc_auc, aps y logloss.
    """
    params = params or PARAMS_LGBM
    idx = [nombres.index(v) for v in vars_sel]

    modelo = lgb.LGBMClassifier(**params)
    modelo.fit(X_train[:, idx], y_train)

    proba = modelo.predict_proba(X_val[:, idx])[:, 1]

    return {
        "n_variables": len(vars_sel),
        "roc_auc": roc_auc_score(y_val, proba),
        "aps": average_precision_score(y_val, proba),
        "logloss": log_loss(y_val, proba),
    }


def comparar_selectores(selectores, k, X_train, y_train, X_val, y_val, nombres):
    """
    Ejecuta varios selectores con el mismo k y devuelve una tabla comparativa.

    Args:
        selectores: dict {etiqueta: callable(X, y, nombres, k) -> list[str]}

    Returns:
        DataFrame indexado por etiqueta de método.
    """
    filas = {}
    for etiqueta, selector in selectores.items():
        vars_sel = selector(X_train, y_train, nombres, k)
        filas[etiqueta] = evaluar_subconjunto(
            vars_sel, X_train, y_train, X_val, y_val, nombres
        )
    return pd.DataFrame(filas).T


def barrido_k(selector, valores_k, X_train, y_train, X_val, y_val, nombres,
              **kwargs):
    """
    Curva rendimiento vs. número de variables para un selector dado.
    Precalcula el ranking una sola vez cuando el selector lo permite.
    """
    registros = []
    for k in valores_k:
        vars_sel = selector(X_train, y_train, nombres, k, **kwargs)
        r = evaluar_subconjunto(vars_sel, X_train, y_train, X_val, y_val, nombres)
        r["k_solicitado"] = k
        registros.append(r)
    return pd.DataFrame(registros)


# ============================================================
# 4. ANÁLISIS DE SOLAPAMIENTO
# ============================================================

def jaccard(a, b):
    """Índice de Jaccard entre dos subconjuntos de variables."""
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def tabla_solapamiento(subconjuntos):
    """
    Matriz de Jaccard entre todos los pares de subconjuntos.

    Args:
        subconjuntos: dict {etiqueta: list[str]}
    """
    etiquetas = list(subconjuntos)
    M = pd.DataFrame(index=etiquetas, columns=etiquetas, dtype=float)
    for a in etiquetas:
        for b in etiquetas:
            M.loc[a, b] = jaccard(subconjuntos[a], subconjuntos[b])
    return M