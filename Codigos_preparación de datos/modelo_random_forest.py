import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                            f1_score, roc_curve, auc, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CARGAR DATASET
# ============================================================
df = pd.read_csv(r'D:/profesores/rriospa/PAE_Cristian/Codigos_preparación de datos/Dataset.csv')
print(f'Dataset cargado: {df.shape}')
print(df['clase_binaria'].value_counts())

X = df.drop(columns=['fecha', 'clase_binaria'])
y = df['clase_binaria']

# ============================================================
# 2. DIVISIÓN TRAIN / TEST
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'\nEntrenamiento: {len(X_train)} píxeles')
print(f'Prueba:        {len(X_test)} píxeles')

# ============================================================
# 3. FUNCIÓN: DICE SCORE
#    Para clasificación binaria Dice == F1, pero lo dejamos
#    explícito para claridad y reporte por fold
# ============================================================
def dice_score(y_true, y_pred):
    """Dice = 2·TP / (2·TP + FP + FN)  — idéntico a F1 binario"""
    return f1_score(y_true, y_pred, average='binary', zero_division=0)

# ============================================================
# 4. CONFIGURAR CANDIDATOS DE HIPERPARÁMETROS
#    Entrenamos varias combinaciones y nos quedamos con la mejor
# ============================================================
candidatos = [
    {'n_estimators': 200, 'max_depth': None,  'min_samples_leaf': 1,  'label': 'depth=None / leaf=1'},
    {'n_estimators': 200, 'max_depth': None,  'min_samples_leaf': 5,  'label': 'depth=None / leaf=5'},
    {'n_estimators': 200, 'max_depth': 20,    'min_samples_leaf': 1,  'label': 'depth=20  / leaf=1'},
    {'n_estimators': 300, 'max_depth': None,  'min_samples_leaf': 3,  'label': 'depth=None / leaf=3 / 300est'},
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

resultados = []   # guardará métricas por candidato
dice_por_fold = {}

print('\n--- Cross-Validation por candidato ---')
for cfg in candidatos:
    modelo_cv = RandomForestClassifier(
        n_estimators = cfg['n_estimators'],
        max_depth    = cfg['max_depth'],
        min_samples_leaf = cfg['min_samples_leaf'],
        class_weight = 'balanced',   # ← NUEVO: corrige desbalance de clases
        random_state = 42,
        n_jobs       = -1
    )

    # cross_validate devuelve métricas por fold
    cv_res = cross_validate(
        modelo_cv, X_train, y_train, cv=cv,
        scoring={'accuracy': 'accuracy', 'f1': 'f1', 'roc_auc': 'roc_auc'},
        return_train_score=True
    )

    dice_folds = cv_res['test_f1']          # Dice == F1 binario
    dice_media = dice_folds.mean()
    dice_std   = dice_folds.std()

    dice_por_fold[cfg['label']] = dice_folds

    resultados.append({
        'label':      cfg['label'],
        'config':     cfg,
        'dice_media': dice_media,
        'dice_std':   dice_std,
        'acc_media':  cv_res['test_accuracy'].mean(),
        'auc_media':  cv_res['test_roc_auc'].mean(),
    })

    print(f"  {cfg['label']:35s} | Dice: {dice_media:.4f} ± {dice_std:.4f} | "
          f"Acc: {cv_res['test_accuracy'].mean():.4f} | AUC: {cv_res['test_roc_auc'].mean():.4f}")

# ============================================================
# 5. SELECCIONAR MEJOR CONFIGURACIÓN (mayor Dice promedio)
# ============================================================
mejor = max(resultados, key=lambda r: r['dice_media'])
print(f'\n✅ Mejor configuración: {mejor["label"]}')
print(f'   Dice CV: {mejor["dice_media"]:.4f} ± {mejor["dice_std"]:.4f}')

cfg_mejor = mejor['config']
modelo_final = RandomForestClassifier(
    n_estimators     = cfg_mejor['n_estimators'],
    max_depth        = cfg_mejor['max_depth'],
    min_samples_leaf = cfg_mejor['min_samples_leaf'],
    class_weight     = 'balanced',
    random_state     = 42,
    n_jobs           = -1
)
modelo_final.fit(X_train, y_train)

# ============================================================
# 6. EVALUACIÓN EN TEST
# ============================================================
y_pred      = modelo_final.predict(X_test)
y_proba     = modelo_final.predict_proba(X_test)[:, 1]

acc         = accuracy_score(y_test, y_pred)
dice_test   = dice_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc     = auc(fpr, tpr)

print('\n=== RESULTADOS MODELO FINAL (test) ===')
print(f'Accuracy : {acc:.4f}')
print(f'Dice Score (F1): {dice_test:.4f}')
print(f'AUC-ROC  : {roc_auc:.4f}')
print('\nReporte de clasificación:')
print(classification_report(y_test, y_pred, target_names=['No urbano', 'Urbano']))

# ============================================================
# 7. VISUALIZACIONES
# ============================================================
sns.set_theme(style='whitegrid', palette='muted')
COLORES = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Random Forest — Clasificación de Coberturas de Suelo',
             fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 7.1 Distribución de clases ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
conteos = y.value_counts()
bars = ax1.bar(['No urbano', 'Urbano'], conteos.values,
               color=COLORES[:2], edgecolor='white', linewidth=1.2)
for bar, v in zip(bars, conteos.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(conteos)*0.01,
             f'{v:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_title('Distribución de clases', fontweight='bold')
ax1.set_ylabel('Número de píxeles')

# ── 7.2 Dice Score por fold (todos los candidatos) ──────────
ax2 = fig.add_subplot(gs[0, 1])
folds = np.arange(1, 6)
for i, (label, folds_dice) in enumerate(dice_por_fold.items()):
    ax2.plot(folds, folds_dice, marker='o', label=label,
             color=COLORES[i % len(COLORES)], linewidth=1.8)
ax2.set_title('Dice Score por fold (CV)', fontweight='bold')
ax2.set_xlabel('Fold')
ax2.set_ylabel('Dice Score')
ax2.set_xticks(folds)
ax2.legend(fontsize=7, loc='lower right')
ax2.set_ylim(0, 1)

# ── 7.3 Comparación de candidatos (barras Dice ± std) ────────
ax3 = fig.add_subplot(gs[0, 2])
labels_c = [r['label'] for r in resultados]
medias   = [r['dice_media'] for r in resultados]
stds     = [r['dice_std']   for r in resultados]
colores_c = [COLORES[2] if r['label'] == mejor['label'] else '#AAAAAA' for r in resultados]
bars3 = ax3.barh(labels_c, medias, xerr=stds, color=colores_c,
                 edgecolor='white', capsize=4)
ax3.set_title('Dice CV promedio por configuración', fontweight='bold')
ax3.set_xlabel('Dice Score')
ax3.set_xlim(0, 1)
ax3.axvline(mejor['dice_media'], color='red', linestyle='--', linewidth=1, alpha=0.6)
for bar, v in zip(bars3, medias):
    ax3.text(v + 0.005, bar.get_y() + bar.get_height()/2,
             f'{v:.4f}', va='center', fontsize=8)

# ── 7.4 Matriz de confusión ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['No urbano', 'Urbano'])
disp.plot(ax=ax4, colorbar=False, cmap='Blues')
ax4.set_title('Matriz de confusión (test)', fontweight='bold')

# ── 7.5 Curva ROC ────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(fpr, tpr, color=COLORES[0], lw=2, label=f'AUC = {roc_auc:.4f}')
ax5.plot([0, 1], [0, 1], 'k--', lw=1)
ax5.fill_between(fpr, tpr, alpha=0.1, color=COLORES[0])
ax5.set_title('Curva ROC', fontweight='bold')
ax5.set_xlabel('Tasa de Falsos Positivos')
ax5.set_ylabel('Tasa de Verdaderos Positivos')
ax5.legend(loc='lower right')
ax5.set_xlim([0, 1])
ax5.set_ylim([0, 1.02])

# ── 7.6 Métricas finales (resumen visual) ────────────────────
ax6 = fig.add_subplot(gs[1, 2])
metricas  = ['Accuracy', 'Dice Score\n(F1)', 'AUC-ROC']
valores   = [acc, dice_test, roc_auc]
colores_m = [COLORES[0] if v >= 0.8 else COLORES[3] for v in valores]
bars6 = ax6.bar(metricas, valores, color=colores_m, edgecolor='white', linewidth=1.2)
for bar, v in zip(bars6, valores):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{v:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax6.set_title('Métricas en conjunto de prueba', fontweight='bold')
ax6.set_ylim(0, 1.15)
ax6.axhline(0.8, color='gray', linestyle=':', linewidth=1)
ax6.text(2.4, 0.81, 'umbral 0.8', fontsize=8, color='gray')

# ── 7.7 Importancia de features (top 20) ────────────────────
ax7 = fig.add_subplot(gs[2, :])
importancias = pd.Series(modelo_final.feature_importances_, index=X.columns)
top20 = importancias.nlargest(20).sort_values()
colores_f = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top20)))
top20.plot(kind='barh', ax=ax7, color=colores_f, edgecolor='white')
ax7.set_title('Top 20 — Importancia de variables (Mean Decrease Impurity)',
              fontweight='bold')
ax7.set_xlabel('Importancia relativa')

plt.savefig(
    r'D:/profesores/rriospa/PAE_Cristian/Codigos_preparación de datos/resultados_RF.png',
    dpi=150, bbox_inches='tight'
)
plt.show()
print('\nGráfico guardado: resultados_RF.png')

# ============================================================
# 8. GUARDAR MEJOR MODELO
# ============================================================
joblib.dump(modelo_final, 
    r'D:/profesores/rriospa/PAE_Cristian/Codigos_preparación de datos/modelo_random_forest.pkl')
print('Modelo guardado: modelo_random_forest.pkl')