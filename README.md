# Scoring-Credit-Risk
Modelo de scoring de riesgo crediticio (mora a 60 días en 12 meses) con XGBoost monótono. Incluye validación out-of-time, tratamiento de desbalanceo (class weights, SMOTE, undersampling), selección de variables por Gini y correlación, explicabilidad (SHAP, DiCE, árbol sustituto) y un simulador web que corre el modelo en el navegador.

# Modelo de riesgo crediticio

Modelo de *scoring* de riesgo de crédito que estima la probabilidad de que un cliente entre en mora (60 días de impago en los 12 meses siguientes: `TARGET_60D_12M`). El proyecto cubre todo el ciclo: selección de variables, tratamiento del desbalanceo, validación temporal, comparación de modelos, explicabilidad y un **simulador web** que ejecuta el modelo entrenado directamente en el navegador.

> ⚠️ **Aviso.** Proyecto con fines formativos. La puntuación es un *score* ordinal, **no** una probabilidad de impago calibrada, y no constituye una decisión de crédito real.

---

## Tabla de contenidos

- [Objetivo](#objetivo)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Metodología](#metodología)
  - [1. Datos y variable objetivo](#1-datos-y-variable-objetivo)
  - [2. Selección de variables](#2-selección-de-variables)
  - [3. Valores faltantes](#3-valores-faltantes)
  - [4. Validación out-of-time](#4-validación-out-of-time)
  - [5. Tratamiento del desbalanceo](#5-tratamiento-del-desbalanceo)
  - [6. Modelos comparados](#6-modelos-comparados)
  - [7. Métricas y umbral de decisión](#7-métricas-y-umbral-de-decisión)
  - [8. Explicabilidad](#8-explicabilidad)
- [Resultados y conclusiones](#resultados-y-conclusiones)
- [El simulador](#el-simulador)
- [Cómo reproducirlo](#cómo-reproducirlo)
- [Requisitos](#requisitos)
- [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)

---

## Objetivo

Predecir qué clientes incurrirán en mora (60 días de impago dentro de una ventana de 12 meses) a partir de variables de comportamiento, saldos, historial en el sistema financiero y uso de productos. El problema es de **clasificación binaria con clases muy desbalanceadas** (~10-11 % de morosos), lo que condiciona casi todas las decisiones del *pipeline*.

---

## Estructura del repositorio

```
.
├── notebook.ipynb            # Pipeline completo: EDA, selección, modelos, explicabilidad
├── exportar_simulador.py     # Celda final de Colab: exporta el modelo a simulador_modelo.json
├── simulador.html            # Simulador web autónomo (corre el modelo en el navegador)
└── README.md
```

---

## Metodología

### 1. Datos y variable objetivo

- **Objetivo:** `TARGET_60D_12M` — 1 si el cliente alcanza 60 días de atraso en los 12 meses siguientes, 0 en caso contrario.
- **Identificadores temporales:** `CODMES` (año-mes) e `ID_CLIENTE`. Se conservan hasta **después** de partir los datos, porque son necesarios para la validación temporal y para evitar fugas entre clientes; nunca entran como *features* al modelo.

### 2. Selección de variables

Dos filtros encadenados:

**a) Poder predictivo univariante (Gini).** Para cada variable se calcula el Gini como `|2·AUC − 1|`, ignorando nulos:

```python
def gini_univariante(x, y):
    mask = x.notna()
    if x[mask].nunique() < 2:
        return 0.0
    return abs(2 * roc_auc_score(y[mask], x[mask]) - 1)
```

**b) Eliminación de redundancia (correlación de Spearman).** Se ordenan las variables por Gini y, de forma *greedy*, se descarta toda variable que supere un umbral de correlación (0.85) con otra ya conservada de mayor poder:

```python
def eliminar_alta_correlacion(df, umbral=0.85, target='TARGET_60D_12M',
                              metodo='spearman', muestra=100_000):
    X = df.drop(columns=[target]); y = df[target]
    poder = pd.Series({c: gini_univariante(X[c], y) for c in X.columns}).sort_values(ascending=False)
    corr = X.sample(min(len(X), muestra), random_state=42).corr(method=metodo).abs()
    conservadas, eliminadas = [], {}
    for var in poder.index:
        redundante_con = [k for k in conservadas if corr.loc[var, k] > umbral]
        if redundante_con: eliminadas[var] = redundante_con[0]
        else: conservadas.append(var)
    return df[conservadas + [target]]
```

> **Matiz importante:** algunas variables con Gini univariante ≈ 0 **sí** aportan al modelo. El Gini solo capta relaciones monótonas; una variable con relación en forma de U (riesgo alto tanto en valores bajos como altos) tiene un AUC ≈ 0.5 y Gini ≈ 0, pero los árboles de XGBoost sí explotan ese patrón. Por eso varias variables de bajo Gini pero tendencia clara por quintiles (edad, antigüedad, nº de entidades) se conservaron manualmente.

### 3. Valores faltantes

- **XGBoost** gestiona los `NaN` de forma nativa: cada nodo aprende hacia qué rama (`missing`) enviar los valores ausentes.
- Los modelos de **scikit-learn** (regresión logística, DiCE, LIME, árbol sustituto) no admiten nulos, así que se imputa con `SimpleImputer(strategy='median')` antes de escalar.

> Esta diferencia es la causa de una incoherencia habitual: un cliente con todos los valores nulos recibe explicaciones distintas en SHAP (que respeta los `NaN`) y en DiCE (que ve las medianas imputadas). Para comparar explicaciones conviene usar clientes sin nulos.

### 4. Validación out-of-time

En lugar de una partición aleatoria, se separa por tiempo: los meses antiguos entrenan y los **tres últimos** validan. Así se mide el modelo como se usará en producción — prediciendo el futuro con el pasado.

```python
meses = sorted(df_train_clean['CODMES'].unique())
corte = meses[-3]
train = df_train_clean[df_train_clean['CODMES'] < corte]
test  = df_train_clean[df_train_clean['CODMES'] >= corte]
X_train = train.drop(columns=['TARGET_60D_12M', 'ID_CLIENTE', 'CODMES'])
y_train = train['TARGET_60D_12M']
X_test  = test.drop(columns=['TARGET_60D_12M', 'ID_CLIENTE', 'CODMES'])
y_test  = test['TARGET_60D_12M']
```

- **Train:** 202403 → 202502 · **Test:** 202503 → 202505
- Tasa de mora: **11.5 % en train vs 10.4 % en test**. Un *split* aleatorio habría ocultado esta deriva temporal.

### 5. Tratamiento del desbalanceo

Se comparan cuatro estrategias para que la clase minoritaria (morosos) no quede diluida:

| Estrategia | Idea |
|---|---|
| Regresión logística + `class_weight='balanced'` | Penaliza más el error en la clase minoritaria |
| `RandomUnderSampler` | Submuestrea la clase mayoritaria |
| `SMOTE` | Genera ejemplos sintéticos de la clase minoritaria |
| XGBoost + `scale_pos_weight` | Reescala el gradiente de la clase positiva |

### 6. Modelos comparados

- **Regresión logística** (línea base interpretable, con imputación + estandarizado).
- **XGBoost con `scale_pos_weight`** (modelo libre, máxima capacidad).
- **XGBoost monótono** — con restricciones de monotonía en las variables donde el signo del efecto es conocido *a priori* (p. ej. más atrasos ⇒ más riesgo). Es el modelo **desplegado**: ligeramente menos potente pero mucho más defendible ante un comité de riesgos, porque no puede contradecir la lógica de negocio.

### 7. Métricas y umbral de decisión

Métricas propias de *scoring* bancario:

- **AUC-ROC** — capacidad de ordenación.
- **Gini** = `2·AUC − 1`.
- **KS** — máxima separación entre las distribuciones acumuladas de buenos y malos.

El **umbral de corte** se fija en el punto KS-óptimo de la curva ROC (máximo `TPR − FPR`):

```python
prob_test = MODELO.predict_proba(X_test)[:, 1]
fpr, tpr, thr = roc_curve(y_test, prob_test)
umbral = float(thr[int(np.argmax(tpr - fpr))])
```

> ⚠️ Usa siempre `predict_proba(...)[:, 1]` (probabilidades), no `predict(...)` (clases 0/1), o el umbral KS saldrá degenerado (p. ej. 1.000).

| Métrica (test) | Valor |
|---|---|
| AUC | _(completar con tus resultados)_ |
| Gini | _(completar)_ |
| KS | _(completar)_ |
| Umbral KS-óptimo | _(completar)_ |

### 8. Explicabilidad

- **SHAP** (`TreeExplainer`): importancia global (*beeswarm*) y descomposición individual (*waterfall*).
- **DiCE**: contrafactuales — qué tendría que cambiar un cliente para cambiar de decisión.
- **PDP / ICE**: efecto marginal de cada variable.
- **LIME**: explicación local aproximada.
- **Árbol sustituto global**: un árbol de decisión poco profundo que imita al modelo (**fidelidad ≈ 84.7 %**), para leer la lógica general de un vistazo.

---

## Resultados y conclusiones

- La **validación temporal** revela una deriva de la mora (11.5 % → 10.4 %) que una partición aleatoria esconde. Es la forma honesta de estimar el rendimiento futuro.
- El **Gini univariante no basta** para descartar variables: capta solo relaciones monótonas. Variables con Gini ≈ 0 pero relación en U resultaron relevantes en SHAP.
- El **XGBoost monótono** es la opción más equilibrada: apenas pierde capacidad frente al libre y gana muchísimo en defensibilidad e interpretabilidad.
- Las técnicas de desbalanceo mueven el equilibrio entre **cazar morosos** y **no rechazar buenos clientes de más**; la elección depende del coste de cada error.

---

## El simulador

`simulador.html` es una página **autónoma** que carga el modelo entrenado y evalúa solicitantes **íntegramente en el navegador**: sin servidor, sin instalar nada, y sin que los datos salgan del equipo. Internamente recorre los árboles de XGBoost en JavaScript.

**Cómo usarlo con tu modelo real:**

1. Pega `exportar_simulador.py` al final de tu notebook y ejecútalo. Descarga un único fichero `simulador_modelo.json` con los árboles, los estadísticos de cada variable, el umbral, las métricas y 50 ejemplos de verificación.
2. Abre `simulador.html` en el navegador y pulsa **«Cargar modelo (.json)»**.
3. El formulario se reconstruye con tus variables (las 8 más importantes visibles, el resto plegadas). Rellena un cliente y pulsa **Evaluar**.

Cada evaluación devuelve el veredicto (aprobado/denegado), el *score* frente al umbral y los factores que más pesan en la decisión, calculados por **ablación** (cuánto mueve el riesgo cada valor respecto al cliente típico, en log-odds). Marca **N/D** en cualquier campo desconocido y se tratará como faltante, igual que en el entrenamiento.

**Por qué es robusto a la versión de XGBoost:** el `base_score` de XGBoost cambia entre versiones y descuadra las probabilidades. En lugar de reconstruirlo, el JSON incluye 50 clientes con su probabilidad exacta, y el simulador deriva el intercepto como `mediana(logit(prob) − suma_de_hojas)`. Una insignia **«verificado ✓»** confirma que las predicciones del navegador coinciden con las del modelo en Colab.

---

## Cómo reproducirlo

1. Abre `notebook.ipynb` en Google Colab o Jupyter.
2. Ejecuta las celdas en orden: carga de datos → selección de variables → *split* out-of-time → entrenamiento y comparación → explicabilidad.
3. Ejecuta `exportar_simulador.py` para generar `simulador_modelo.json`.
4. Abre `simulador.html` y carga ese fichero.

---

## Requisitos

```
python >= 3.9
pandas
numpy
scikit-learn
xgboost
imbalanced-learn   # SMOTE, RandomUnderSampler
shap
dice-ml            # contrafactuales
lime
matplotlib
```

El simulador **no requiere nada**: es HTML+JS puro que se abre en cualquier navegador.

---

## Limitaciones y trabajo futuro

- El *score* es **ordinal**, no una probabilidad de impago calibrada (el `scale_pos_weight` la infla). Para mostrar una PD real habría que calibrar con `CalibratedClassifierCV` y reexportar.
- La selección de variables usa correlación lineal/monótona; relaciones no lineales entre pares de variables no se evalúan en el filtro de redundancia.
- No hay seguimiento de estabilidad del modelo en el tiempo (PSI, *backtesting* mensual).
- Falta un análisis de equidad/sesgo por grupos.
