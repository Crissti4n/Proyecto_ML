# Procesamiento de datos espacio-temporales para el modelamiento del crecimiento urbano en Valledupar

Este proyecto implementa un pipeline de clasificación supervisada para la
identificación binaria de cobertura urbana y no urbana en la ciudad de
**Valledupar, Colombia**, a partir de imágenes satelitales multitemporales
Landsat del periodo **1993–2003**. El trabajo hace parte de una tesis de
grado desarrollada en la **Universidad Nacional de Colombia**.
---

## Descripción del pipeline

### 1. `CSV_IMG_satelitales.py` — Construcción del dataset

Procesa 20 pares de imágenes multiespectrales y clasificadas en formato `.img`
(ERDAS IMAGINE) correspondientes al periodo 1993–2003. De cada píxel válido
se extraen los siguientes atributos:

- Bandas espectrales: B1, B2, B3, B4, B5 y B7 (Landsat 5 y 8)
- Índices espectrales: NDVI, NDWI, EVI y NDBI
- Coordenadas geográficas: latitud y longitud, obtenidas mediante transformación
  de proyección UTM zona 18N a WGS84
- Etiqueta binaria: 1 = urbano, 0 = no urbano

La asignación de la etiqueta urbana se realiza comparando el valor de clase de
cada píxel contra el valor digital (DN) característico de la clase urbana
definida durante la clasificación supervisada manual en ERDAS IMAGINE. Dado que
las muestras de entrenamiento se seleccionaron de forma independiente para cada
escena, este valor de referencia varía entre imágenes, reflejando las diferencias
espectrales inherentes a cada fecha de adquisición.

Salida: `Dataset.csv`, con una observación por píxel en formato tabular.

---

### 2. `modelo_random_forest.py` — Entrenamiento y evaluación del modelo

Entrena un clasificador **Random Forest** sobre `Dataset.csv` siguiendo el
flujo de trabajo descrito a continuación:

- División estratificada 80/20 (entrenamiento/prueba), preservando la
  proporción de clases en ambas particiones
- Búsqueda de hiperparámetros mediante validación cruzada estratificada de
  cinco pliegues (StratifiedKFold, k=5), utilizando el Dice Score (F1 binario)
  como criterio de selección
- Corrección del desbalance de clases mediante el parámetro
  `class_weight='balanced'`

#### Configuraciones de hiperparámetros evaluadas

| Configuración       | n_estimators | max_depth | min_samples_leaf |
|---------------------|:------------:|:---------:|:----------------:|
| None / leaf=1       | 200          | None      | 1                |
| None / leaf=5       | 200          | None      | 5                |
| depth=20 / leaf=1   | 200          | 20        | 1                |
| **None / leaf=3**   | **300**      | **None**  | **3**            |

La configuración resaltada fue la seleccionada por obtener el mayor Dice Score
promedio en validación cruzada (0.9310).

#### Resultados sobre el conjunto de prueba

| Métrica    | Valor  |
|------------|:------:|
| Accuracy   | 0.9708 |
| Dice Score | 0.9334 |
| AUC-ROC    | 0.9952 |

Salida: `modelo_random_forest.pkl`, modelo serializado listo para su uso.

---

### 3. `clasificación_binaria_forest.py` — Algoritmo de clasificación espacial

Aplica el modelo entrenado de forma interactiva sobre cualquier escena del
dataset siguiendo estas etapas:

1. Carga del modelo entrenado y el dataset completo en formato tabular
2. El usuario selecciona una escena por fecha a partir de la lista de fechas
   disponibles
3. El modelo genera por píxel la etiqueta de clase predicha y la probabilidad
   asociada a la clase urbana
4. Se reconstruye la cuadrícula ráster original mediante transformación inversa
   de coordenadas (WGS84 a UTM zona 18N, luego a fila y columna)
5. El resultado se exporta como un GeoTIFF de dos bandas: banda 1 con la
   clasificación binaria y banda 2 con el mapa de probabilidad continua
6. Se generan y guardan visualizaciones en alta resolución junto con el
   porcentaje de área urbana detectada para la escena procesada

---

## Descarga de datos

El dataset de entrenamiento (`Dataset.csv`) y el modelo entrenado
(`modelo_random_forest.pkl`) están disponibles para su descarga en el
siguiente enlace:

[Google Drive — Dataset y modelo entrenado](https://drive.google.com/drive/folders/1qx1OtWYnESgWxZnhQYqNrw_dE47207Jt?usp=drive_link)

---

## Requisitos

```bash
pip install rasterio pyproj pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## Limitaciones conocidas y trabajo futuro

La calidad de la clasificación supervisada manual realizada en ERDAS IMAGINE
incide directamente en el desempeño del modelo, ya que los errores introducidos
durante la generación de firmas espectrales se propagan al dataset de
entrenamiento. Una revisión más rigurosa de las regiones de interés (ROIs)
seleccionadas por escena, orientada a reducir el ruido espectral entre clases,
representa la mejora de mayor impacto potencial sobre la precisión del
clasificador.

Por otro lado, la extracción a nivel de píxel genera un volumen de observaciones
muy elevado, lo que representa un desafío computacional a medida que se
incorporan más imágenes a la serie temporal. La aplicación de técnicas de
reducción de dimensionalidad, como el Análisis de Componentes Principales (PCA)
o la selección de características basada en importancia de variables, permitiría
mejorar la escalabilidad del pipeline hacia series de mayor extensión temporal
sin sacrificar la capacidad discriminativa del modelo.

Los resultados presentados corresponden a una prueba de concepto sobre un
subconjunto acotado de datos (1993–2003). La extensión del análisis a la
totalidad de la serie temporal disponible (1984–2026) está prevista como
trabajo futuro.

---

## Autor

Cristian Hernandez  
Universidad Nacional de Colombia  
chernandezos@unal.edu.co  
ORCID: https://orcid.org/0009-0000-0477-9822
