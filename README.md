# TFM---Baterias

## ANÁLISIS Y PREDICCIÓN DE MATERIALES PARA BATERÍAS MEDIANTE APRENDIZAJE AUTOMÁTICO

Este repositorio contiene el flujo completo desarrollado para el TFM orientado al análisis, modelado y selección de materiales candidatos para electrodos de baterías a partir de datos públicos de **Materials Project**.

El proyecto combina:

- extracción de datos mediante la API de Materials Project;
- inspección, limpieza y análisis exploratorio;
- ingeniería de características composicionales con descriptores Magpie;
- modelos clásicos de machine learning para predecir `average_voltage`;
- estudio específico del target `max_delta_volume`;
- modelado estructural mediante redes neuronales de grafos (GNN);
- pipeline final de inferencia y screening de nuevos candidatos;
- análisis de interpretabilidad del modelo de voltaje.

---

## 1. Estructura de notebooks

Los notebooks deben ejecutarse siguiendo, en general, este orden:

```text
00_extraccion_datos_materials_project.ipynb
        ↓
01_inspeccion_y_limpieza.ipynb
        ↓
02_feature engineering.ipynb
        ↓
03_train_test_split.ipynb
        ↓
04_modelado.ipynb
        ↓
05_dataset GNN.ipynb
        ↓
06_modelado volumen parte 1.ipynb
        ↓
07_modelado volume parte 2.ipynb
        ↓
08_pipeline_final_screening.ipynb
        ↓
09_interpretabilidad_modelo_voltaje.ipynb
```

### `00_extraccion_datos_materials_project.ipynb`

Descarga los datos de electrodos de inserción desde Materials Project utilizando `MPRester`.

Genera los archivos iniciales dentro de `datos_baterias_mp/`, entre ellos:

```text
datos_baterias_mp/
├── baterias_materials_project.jsonl
└── baterias_materials_project_resumen.csv
```

Este notebook requiere una API key de Materials Project almacenada en la variable de entorno `MP_API_KEY`.

### `01_inspeccion_y_limpieza.ipynb`

Realiza la inspección inicial, análisis exploratorio, tratamiento de valores problemáticos y limpieza del dataset.

El resultado utilizado posteriormente es:

```text
datos_baterias_mp/baterias_clean.csv
```

### `02_feature engineering.ipynb`

Construye las variables de entrada para los modelos tabulares.

Las características principales se obtienen a partir de:

- composición del framework;
- descriptores composicionales **Magpie** mediante `matminer`;
- codificación del `working_ion`.

Genera:

```text
datos_baterias_mp/baterias_features_engineered.csv
```

### `03_train_test_split.ipynb`

Separa los datos utilizados por los modelos tabulares en conjuntos de entrenamiento y test.

Entre los archivos generados se encuentran:

```text
datos_baterias_mp/
├── X_train.csv
├── X_test.csv
├── y_voltage_train.csv
├── y_voltage_test.csv
├── y_volume_train.csv
└── y_volume_test.csv
```

### `04_modelado.ipynb`

Compara distintos modelos para la predicción de `average_voltage`:

- Dummy Regressor;
- Ridge Regression;
- Random Forest;
- XGBoost.

El modelo finalmente utilizado en el pipeline de screening es XGBoost.

Para poder ejecutar el notebook 07, el modelo seleccionado debe guardarse como:

```text
modelos_finales/xgboost_voltage.json
```
### `05_dataset GNN.ipynb`
El notebook 05_dataset_GNN.ipynb constituye el paso de preparación específico para el modelado mediante redes neuronales de grafos. En él se construye el dataset estructural utilizado por los modelos GNN y se guarda en:
```text
datos_baterias_mp/gnn_dataset.pkl
```
Este archivo contiene, además de las variables procedentes del dataset de baterías, las estructuras cristalinas necesarias para representar cada material como un grafo.

### `06_modelado volumen parte 1.ipynb`

Inicia el estudio específico de `max_delta_volume` y desarrolla las primeras redes neuronales de grafos.

Las estructuras cristalinas se representan como grafos en los que:

- los nodos representan átomos;
- las características de los nodos contienen propiedades atómicas;
- las aristas representan vecindades periódicas;
- las distancias interatómicas pueden incorporarse como características de las aristas.

### `07_modelado volume parte 2.ipynb`

Desarrolla y compara distintas arquitecturas GNN para la predicción de `max_delta_volume`, incluyendo modelos basados en `GCNConv` y `GINEConv`.

El modelo estructural seleccionado para el pipeline final es `GINE_distance`.

Para ejecutar el notebook 07 debe existir:

```text
modelos_finales/gine_distance_volume.pt
```

### `08_pipeline_final_screening.ipynb`

Implementa el pipeline final de inferencia y screening.

El flujo general es:

```text
Materials Project
        ↓
obtención de candidatos
        ↓
identificación del working ion / framework
        ↓
features Magpie
        ↓
XGBoost → predicción de average_voltage
        ↓
estructura cristalina → grafo
        ↓
GINE → predicción de max_delta_volume
        ↓
controles de calidad y estabilidad
        ↓
ranking
        ↓
shortlist final
```

El pipeline utiliza las carpetas:

```text
datos_baterias_mp/
modelos_finales/
resultados_screening/
```

Los principales resultados se exportan como:

```text
resultados_screening/
├── screening_predicciones.csv
└── candidatos_finales.csv
```

El screening final se restringe a working ions suficientemente representados en el dataset de entrenamiento y aplica controles para evitar predicciones de voltaje excesivamente alejadas del dominio observado.

La predicción de cambio de volumen se mantiene como información del modelo estructural, aunque debe interpretarse con precaución en nuevos candidatos debido a la fuerte concentración observada en determinadas predicciones.

### `09_interpretabilidad_modelo_voltaje.ipynb`

Analiza el comportamiento del modelo XGBoost de voltaje mediante técnicas clásicas de interpretabilidad:

- importancia nativa de XGBoost;
- permutation importance;
- comparación entre ambas medidas;
- importancia agregada por familias de variables;
- análisis de residuos y errores;
- estudio de los casos con mayor error.

Los resultados se guardan en:

```text
resultados_interpretabilidad/
```
## 2. Instalación
Crear un entorno virtual
En Windows PowerShell:
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```
En Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```
Actualizar pip
```bash
python -m pip install --upgrade pip
```
Instalar las dependencias principales
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost pymatgen matminer mp-api typing-extensions jupyter ipykernel
```
Instalar PyTorch y PyTorch Geometric
El modelado GNN requiere además:
```bash
pip install torch torch-geometric
```
---
## 3. API de Materials Project
Los notebooks de extracción y screening acceden a Materials Project mediante `mp-api`.
La API key no debe escribirse directamente dentro de los notebooks ni subirse al repositorio.
El código espera encontrarla en:
```python
os.environ\["MP\_API\_KEY"]
```
Configurar la variable en Windows PowerShell
```powershell
\[System.Environment]::SetEnvironmentVariable(
    "MP\_API\_KEY",
    "TU\_API\_KEY",
    "User"
)
```
Después de crear la variable, reiniciar VS Code/Jupyter para que el nuevo proceso pueda leerla.
Para comprobarla sin mostrar su contenido:
```python
import os

print("MP\_API\_KEY configurada:", bool(os.environ.get("MP\_API\_KEY")))
```
---
## 4. Estructura recomendada del proyecto
```text
TFM/
│
├── 00\_extraccion\_datos\_materials\_project.ipynb
├── 01\_inspeccion\_y\_limpieza.ipynb
├── 02\_feature engineering.ipynb
├── 03\_train\_test\_split.ipynb
├── 04\_modelado.ipynb
├── 05\_modelado volumen parte 1.ipynb
├── 06\_modelado volume parte 2.ipynb
├── 07\_pipeline\_final\_screening.ipynb
├── 08\_interpretabilidad\_modelo\_voltaje.ipynb
│
├── datos\_baterias\_mp/
│   ├── baterias\_materials\_project.jsonl
│   ├── baterias\_materials\_project\_resumen.csv
│   ├── baterias\_clean.csv
│   ├── baterias\_features\_engineered.csv
│   ├── X\_train.csv
│   ├── X\_test.csv
│   ├── y\_voltage\_train.csv
│   ├── y\_voltage\_test.csv
│   ├── y\_volume\_train.csv
│   └── y\_volume\_test.csv
│
├── modelos\_finales/
│   ├── xgboost\_voltage.json
│   └── gine\_distance\_volume.pt
│
├── resultados\_screening/
│   ├── screening\_predicciones.csv
│   └── candidatos\_finales.csv
│
└── resultados\_interpretabilidad/
```
---
## 5. Ejecución
La forma más segura de reproducir el proyecto es ejecutar los notebooks en orden desde `00` hasta `08`.
Para abrir Jupyter:
```bash
jupyter notebook
```
También pueden ejecutarse directamente desde VS Code seleccionando como kernel el entorno virtual en el que se instalaron las dependencias.

### Ejecución mínima del screening final
Si los datos ya han sido procesados y los modelos ya están entrenados, el notebook `07\_pipeline\_final\_screening.ipynb` puede ejecutarse directamente siempre que existan, como mínimo:
```text
modelos\_finales/xgboost\_voltage.json
modelos\_finales/gine\_distance\_volume.pt
datos\_baterias\_mp/X\_train.csv
datos\_baterias\_mp/baterias\_clean.csv
```
y esté configurada la variable de entorno:
```text
MP\_API\_KEY
```


