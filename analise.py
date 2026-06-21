# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11

# =============================================================================
# FUNCOES UTILITARIAS
# =============================================================================

def avaliar_modelo(nome, modelo, X_train, X_test, y_train, y_test, classes):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    print("\n" + "="*55)
    print("  " + nome)
    print("="*55)
    print("  Acuracia : {:.4f} ({:.2f}%)".format(acc, acc*100))
    print("  F1-Score : {:.4f}".format(f1))
    print()
    print(classification_report(y_test, y_pred, target_names=classes))
    return acc, f1, y_pred


def plot_confusion(nome, y_test, y_pred, classes, ax):
    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(nome, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)


def plot_comparacao(nomes, acuracias, f1s, titulo, ax1, ax2):
    x = np.arange(len(nomes))
    cores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    ax1.bar(x, acuracias, color=cores[:len(nomes)])
    ax1.set_xticks(x)
    ax1.set_xticklabels(nomes, rotation=15)
    ax1.set_ylim(0, 1)
    ax1.set_title(titulo + " - Acuracia")
    ax1.set_ylabel("Acuracia")
    for i, v in enumerate(acuracias):
        ax1.text(i, v + 0.01, "{:.3f}".format(v), ha="center", fontsize=9)

    ax2.bar(x, f1s, color=cores[:len(nomes)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(nomes, rotation=15)
    ax2.set_ylim(0, 1)
    ax2.set_title(titulo + " - F1-Score")
    ax2.set_ylabel("F1-Score (weighted)")
    for i, v in enumerate(f1s):
        ax2.text(i, v + 0.01, "{:.3f}".format(v), ha="center", fontsize=9)


# =============================================================================
# DATASET 1 - OBESITY LEVELS
# =============================================================================
print("\n" + "#"*65)
print("#  DATASET 1: ESTIMATION OF OBESITY LEVELS")
print("#"*65)

df_ob = pd.read_csv(
    os.path.join(BASE_DIR, "datasets", "obesity", "ObesityDataSet_raw_and_data_sinthetic.csv")
)

print("\nFormato: {}".format(df_ob.shape))
print("Colunas: {}".format(list(df_ob.columns)))
print("\nDistribuicao da variavel alvo:")
print(df_ob["NObeyesdad"].value_counts())

# --- Pre-processamento ---
le = LabelEncoder()
df_ob_enc = df_ob.copy()
for col in df_ob_enc.select_dtypes(include="object").columns:
    df_ob_enc[col] = le.fit_transform(df_ob_enc[col])

X_ob = df_ob_enc.drop("NObeyesdad", axis=1)
y_ob = df_ob_enc["NObeyesdad"]
label_enc_ob = LabelEncoder().fit(df_ob["NObeyesdad"])

X_ob_train, X_ob_test, y_ob_train, y_ob_test = train_test_split(
    X_ob, y_ob, test_size=0.2, random_state=42, stratify=y_ob
)

# ---- Modelo 1A: Naive Bayes (obrigatorio) ----
print("\n\n>>> MODELO 1A: NAIVE BAYES (obrigatorio)")
nb_base = GaussianNB()
acc_nb, f1_nb, pred_nb = avaliar_modelo(
    "Naive Bayes (var_smoothing=1e-9)",
    nb_base, X_ob_train, X_ob_test, y_ob_train, y_ob_test,
    label_enc_ob.classes_
)

# --- Variacoes de hiperparametro: var_smoothing ---
print("\n>>> VARIACOES NAIVE BAYES - var_smoothing")
smoothing_vals = [1e-12, 1e-9, 1e-6, 1e-3, 1e-1]
nb_acc_list, nb_f1_list = [], []
for vs in smoothing_vals:
    m = GaussianNB(var_smoothing=vs)
    m.fit(X_ob_train, y_ob_train)
    yp = m.predict(X_ob_test)
    a = accuracy_score(y_ob_test, yp)
    f = f1_score(y_ob_test, yp, average="weighted")
    nb_acc_list.append(a)
    nb_f1_list.append(f)
    print("  var_smoothing={:.0e}  ->  Acc={:.4f}  F1={:.4f}".format(vs, a, f))

# ---- Modelo 1B: Arvore de Decisao (escolhido) ----
print("\n\n>>> MODELO 1B: ARVORE DE DECISAO (escolhido)")
dt_base = DecisionTreeClassifier(random_state=42)
acc_dt, f1_dt, pred_dt = avaliar_modelo(
    "Arvore de Decisao (padrao)",
    dt_base, X_ob_train, X_ob_test, y_ob_train, y_ob_test,
    label_enc_ob.classes_
)

# --- Variacoes de hiperparametro: max_depth e criterion ---
print("\n>>> VARIACOES ARVORE DE DECISAO - max_depth + criterion")
dt_configs = [
    {"max_depth": 3,    "criterion": "gini"},
    {"max_depth": 5,    "criterion": "gini"},
    {"max_depth": 10,   "criterion": "gini"},
    {"max_depth": None, "criterion": "gini"},
    {"max_depth": 5,    "criterion": "entropy"},
    {"max_depth": 10,   "criterion": "entropy"},
]
dt_acc_list, dt_f1_list, dt_labels = [], [], []
for cfg in dt_configs:
    m = DecisionTreeClassifier(random_state=42, **cfg)
    m.fit(X_ob_train, y_ob_train)
    yp = m.predict(X_ob_test)
    a = accuracy_score(y_ob_test, yp)
    f = f1_score(y_ob_test, yp, average="weighted")
    lbl = "depth={} {}".format(cfg["max_depth"], cfg["criterion"][:4])
    dt_acc_list.append(a)
    dt_f1_list.append(f)
    dt_labels.append(lbl)
    print("  {:<28}  ->  Acc={:.4f}  F1={:.4f}".format(lbl, a, f))

# =============================================================================
# DATASET 2 - STUDENTS DROPOUT
# =============================================================================
print("\n\n" + "#"*65)
print("#  DATASET 2: STUDENTS DROPOUT AND ACADEMIC SUCCESS")
print("#"*65)

df_dr = pd.read_csv(
    os.path.join(BASE_DIR, "datasets", "dropout", "data.csv"), sep=";"
)

print("\nFormato: {}".format(df_dr.shape))
print("\nDistribuicao da variavel alvo:")
print(df_dr["Target"].value_counts())

# --- Pre-processamento ---
X_dr = df_dr.drop("Target", axis=1)
y_dr_raw = df_dr["Target"]
le_dr = LabelEncoder()
y_dr = le_dr.fit_transform(y_dr_raw)
classes_dr = le_dr.classes_

scaler = StandardScaler()
X_dr_scaled = scaler.fit_transform(X_dr)

X_dr_train, X_dr_test, y_dr_train, y_dr_test = train_test_split(
    X_dr_scaled, y_dr, test_size=0.2, random_state=42, stratify=y_dr
)

# ---- Modelo 2A: Regressao Logistica (obrigatorio) ----
print("\n\n>>> MODELO 2A: REGRESSAO LOGISTICA (obrigatorio)")
rl_base = LogisticRegression(max_iter=1000, random_state=42)
acc_rl, f1_rl, pred_rl = avaliar_modelo(
    "Regressao Logistica (C=1.0, lbfgs)",
    rl_base, X_dr_train, X_dr_test, y_dr_train, y_dr_test,
    classes_dr
)

# --- Variacoes de hiperparametro: C e solver ---
print("\n>>> VARIACOES REGRESSAO LOGISTICA - C (regularizacao) + solver")
rl_configs = [
    {"C": 0.01,  "solver": "lbfgs"},
    {"C": 0.1,   "solver": "lbfgs"},
    {"C": 1.0,   "solver": "lbfgs"},
    {"C": 10.0,  "solver": "lbfgs"},
    {"C": 1.0,   "solver": "saga"},
    {"C": 10.0,  "solver": "saga"},
]
rl_acc_list, rl_f1_list, rl_labels = [], [], []
for cfg in rl_configs:
    m = LogisticRegression(max_iter=2000, random_state=42, **cfg)
    m.fit(X_dr_train, y_dr_train)
    yp = m.predict(X_dr_test)
    a = accuracy_score(y_dr_test, yp)
    f = f1_score(y_dr_test, yp, average="weighted")
    lbl = "C={} {}".format(cfg["C"], cfg["solver"])
    rl_acc_list.append(a)
    rl_f1_list.append(f)
    rl_labels.append(lbl)
    print("  {:<22}  ->  Acc={:.4f}  F1={:.4f}".format(lbl, a, f))

# ---- Modelo 2B: Redes Neurais MLP (escolhido) ----
print("\n\n>>> MODELO 2B: REDES NEURAIS MLP (escolhido)")
mlp_base = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
acc_mlp, f1_mlp, pred_mlp = avaliar_modelo(
    "MLP (1 camada oculta: 100 neuronios)",
    mlp_base, X_dr_train, X_dr_test, y_dr_train, y_dr_test,
    classes_dr
)

# --- Variacoes de hiperparametro: arquitetura e learning rate ---
print("\n>>> VARIACOES MLP - arquitetura + learning_rate_init")
mlp_configs = [
    {"hidden_layer_sizes": (50,),       "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (100,),      "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (200,),      "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (100, 50),   "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (100,),      "learning_rate_init": 0.01},
    {"hidden_layer_sizes": (100,),      "learning_rate_init": 0.0001},
]
mlp_acc_list, mlp_f1_list, mlp_labels = [], [], []
for cfg in mlp_configs:
    m = MLPClassifier(max_iter=300, random_state=42, **cfg)
    m.fit(X_dr_train, y_dr_train)
    yp = m.predict(X_dr_test)
    a = accuracy_score(y_dr_test, yp)
    f = f1_score(y_dr_test, yp, average="weighted")
    lbl = "{} lr={}".format(cfg["hidden_layer_sizes"], cfg["learning_rate_init"])
    mlp_acc_list.append(a)
    mlp_f1_list.append(f)
    mlp_labels.append(lbl)
    print("  {:<30}  ->  Acc={:.4f}  F1={:.4f}".format(lbl, a, f))

# =============================================================================
# GRAFICOS
# =============================================================================
print("\n\nGerando graficos...")

# --- Fig 1: Matrizes de Confusao ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Matrizes de Confusao - Todos os Modelos", fontsize=14, fontweight="bold")
plot_confusion("Naive Bayes - Obesity",
               y_ob_test, pred_nb, label_enc_ob.classes_, axes[0][0])
plot_confusion("Arvore de Decisao - Obesity",
               y_ob_test, pred_dt, label_enc_ob.classes_, axes[0][1])
plot_confusion("Regressao Logistica - Dropout",
               y_dr_test, pred_rl, classes_dr, axes[1][0])
plot_confusion("MLP - Dropout",
               y_dr_test, pred_mlp, classes_dr, axes[1][1])
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "fig_confusion_matrices.png"), bbox_inches="tight")
plt.close()

# --- Fig 2: Comparacao entre modelos ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Comparacao de Desempenho entre Modelos", fontsize=14, fontweight="bold")
plot_comparacao(["Naive Bayes", "Arvore Decisao"],
                [acc_nb, acc_dt], [f1_nb, f1_dt],
                "Obesity Levels", axes[0][0], axes[0][1])
plot_comparacao(["Reg. Logistica", "MLP"],
                [acc_rl, acc_mlp], [f1_rl, f1_mlp],
                "Students Dropout", axes[1][0], axes[1][1])
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "fig_comparacao_modelos.png"), bbox_inches="tight")
plt.close()

# --- Fig 3: Impacto var_smoothing no Naive Bayes ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Naive Bayes - Impacto do var_smoothing", fontsize=13, fontweight="bold")
labels_vs = ["{:.0e}".format(v) for v in smoothing_vals]
ax1.plot(labels_vs, nb_acc_list, marker="o", color="#4C72B0")
ax1.set_title("Acuracia"); ax1.set_xlabel("var_smoothing"); ax1.set_ylim(0.5, 1.0)
ax2.plot(labels_vs, nb_f1_list, marker="o", color="#DD8452")
ax2.set_title("F1-Score"); ax2.set_xlabel("var_smoothing"); ax2.set_ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "fig_nb_smoothing.png"), bbox_inches="tight")
plt.close()

# --- Fig 4: Impacto hiperparametros na Arvore de Decisao ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Arvore de Decisao - Impacto dos Hiperparametros", fontsize=13, fontweight="bold")
x = np.arange(len(dt_labels))
ax1.bar(x, dt_acc_list, color="#55A868")
ax1.set_xticks(x); ax1.set_xticklabels(dt_labels, rotation=30, ha="right")
ax1.set_title("Acuracia"); ax1.set_ylim(0.8, 1.0)
for i, v in enumerate(dt_acc_list):
    ax1.text(i, v + 0.002, "{:.3f}".format(v), ha="center", fontsize=8)
ax2.bar(x, dt_f1_list, color="#C44E52")
ax2.set_xticks(x); ax2.set_xticklabels(dt_labels, rotation=30, ha="right")
ax2.set_title("F1-Score"); ax2.set_ylim(0.8, 1.0)
for i, v in enumerate(dt_f1_list):
    ax2.text(i, v + 0.002, "{:.3f}".format(v), ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "fig_dt_hiperparametros.png"), bbox_inches="tight")
plt.close()

# --- Fig 5: Impacto hiperparametros na Regressao Logistica ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Regressao Logistica - Impacto dos Hiperparametros", fontsize=13, fontweight="bold")
x = np.arange(len(rl_labels))
ax1.bar(x, rl_acc_list, color="#4C72B0")
ax1.set_xticks(x); ax1.set_xticklabels(rl_labels, rotation=30, ha="right")
ax1.set_title("Acuracia"); ax1.set_ylim(0.7, 1.0)
for i, v in enumerate(rl_acc_list):
    ax1.text(i, v + 0.005, "{:.3f}".format(v), ha="center", fontsize=8)
ax2.bar(x, rl_f1_list, color="#DD8452")
ax2.set_xticks(x); ax2.set_xticklabels(rl_labels, rotation=30, ha="right")
ax2.set_title("F1-Score"); ax2.set_ylim(0.7, 1.0)
for i, v in enumerate(rl_f1_list):
    ax2.text(i, v + 0.005, "{:.3f}".format(v), ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "fig_rl_hiperparametros.png"), bbox_inches="tight")
plt.close()

# --- Fig 6: Impacto hiperparametros no MLP ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("MLP - Impacto dos Hiperparametros", fontsize=13, fontweight="bold")
x = np.arange(len(mlp_labels))
ax1.bar(x, mlp_acc_list, color="#55A868")
ax1.set_xticks(x); ax1.set_xticklabels(mlp_labels, rotation=30, ha="right")
ax1.set_title("Acuracia"); ax1.set_ylim(0.7, 1.0)
for i, v in enumerate(mlp_acc_list):
    ax1.text(i, v + 0.005, "{:.3f}".format(v), ha="center", fontsize=8)
ax2.bar(x, mlp_f1_list, color="#C44E52")
ax2.set_xticks(x); ax2.set_xticklabels(mlp_labels, rotation=30, ha="right")
ax2.set_title("F1-Score"); ax2.set_ylim(0.7, 1.0)
for i, v in enumerate(mlp_f1_list):
    ax2.text(i, v + 0.005, "{:.3f}".format(v), ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "fig_mlp_hiperparametros.png"), bbox_inches="tight")
plt.close()

# --- Fig 7: Distribuicao das classes ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribuicao das Classes nos Datasets", fontsize=13, fontweight="bold")
df_ob["NObeyesdad"].value_counts().plot(kind="bar", ax=ax1, color="#4C72B0", edgecolor="white")
ax1.set_title("Obesity Levels"); ax1.set_xlabel("")
ax1.tick_params(axis="x", rotation=30)
df_dr["Target"].value_counts().plot(kind="bar", ax=ax2, color="#DD8452", edgecolor="white")
ax2.set_title("Students Dropout"); ax2.set_xlabel("")
ax2.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "fig_distribuicao_classes.png"), bbox_inches="tight")
plt.close()

# =============================================================================
# RESUMO FINAL
# =============================================================================
print("\n" + "="*65)
print("RESUMO FINAL")
print("="*65)
print("\nDataset Obesity Levels:")
print("  Naive Bayes       ->  Acc={:.4f}  F1={:.4f}".format(acc_nb, f1_nb))
print("  Arvore de Decisao ->  Acc={:.4f}  F1={:.4f}".format(acc_dt, f1_dt))
print("\nDataset Students Dropout:")
print("  Reg. Logistica    ->  Acc={:.4f}  F1={:.4f}".format(acc_rl, f1_rl))
print("  MLP               ->  Acc={:.4f}  F1={:.4f}".format(acc_mlp, f1_mlp))
print("="*65)
print("\nGraficos salvos em: " + BASE_DIR)
