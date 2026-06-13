import rasterio
import rasterio.transform
import numpy as np
import pandas as pd
from pyproj import Transformer

# --- Rutas de las carpetas ---
ruta_multi = r'D:/profesores/rriospa/PAE_Cristian/Área multiespectral/'
ruta_clas  = r'D:/profesores/rriospa/PAE_Cristian/Clasificación supervisada_ERDAS/'

# --- Transformador UTM zona 18N → WGS84 (se crea una sola vez) ---
transformer = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)

# --- Pares de imágenes con su valor de Urbano ---
imagenes = [
    {'fecha': '19930208', 'urbano': 32},
    {'fecha': '19950910', 'urbano': 26},
    {'fecha': '19951215', 'urbano': 35},
    {'fecha': '19960116', 'urbano': 34},
    {'fecha': '19970118', 'urbano': 33},
    {'fecha': '19970307', 'urbano': 38},
    {'fecha': '19980121', 'urbano': 22},
    {'fecha': '19980310', 'urbano': 26},
    {'fecha': '19980630', 'urbano': 29},
    {'fecha': '19980817', 'urbano': 30},
    {'fecha': '19981223', 'urbano': 9},
    {'fecha': '19990225', 'urbano': 47},
    {'fecha': '19990723', 'urbano': 41},
    {'fecha': '19990820', 'urbano': 37},
    {'fecha': '19990921', 'urbano': 31},
    {'fecha': '20000127', 'urbano': 32},
    {'fecha': '20010708', 'urbano': 33},
    {'fecha': '20020217', 'urbano': 31},
    {'fecha': '20020601', 'urbano': 35},
    {'fecha': '20031213', 'urbano': 26}
]

todos = []

for img in imagenes:
    fecha  = img['fecha']
    urbano = img['urbano']

    ruta_a = ruta_multi + f'A_{fecha}.img'
    ruta_c = ruta_clas  + f'C_{fecha}.img'

    print(f'Procesando {fecha}...')

    with rasterio.open(ruta_a) as src:
        B1 = src.read(1).astype(float)
        B2 = src.read(2).astype(float)
        B3 = src.read(3).astype(float)
        B4 = src.read(4).astype(float)
        B5 = src.read(5).astype(float)
        B7 = src.read(6).astype(float)

        # ── NUEVO: calcular coordenadas de cada píxel ──────────────────
        altura, anchura = B1.shape

        # Índices de fila y columna para cada píxel
        filas = np.arange(altura)
        cols  = np.arange(anchura)
        cols_grid, filas_grid = np.meshgrid(cols, filas)

        # Coordenadas del centro de cada píxel en UTM (metros)
        x_utm, y_utm = rasterio.transform.xy(
            src.transform,
            filas_grid.ravel(),   # filas
            cols_grid.ravel(),    # columnas
            offset='center'       # centro del píxel, no esquina
        )

        # Convertir UTM → Latitud / Longitud
        lon, lat = transformer.transform(x_utm, y_utm)
        # ───────────────────────────────────────────────────────────────

    with rasterio.open(ruta_c) as src:
        clases = src.read(1)

    # Calcular índices espectrales
    ndvi = (B5 - B4) / (B5 + B4 + 1e-10)
    ndwi = (B3 - B5) / (B3 + B5 + 1e-10)
    evi  = 2.5 * (B5 - B4) / (B5 + 6*B4 - 7.5*B1 + 1 + 1e-10)
    ndbi = (B5 - B4) / (B5 + B4 + 1e-10)

    df = pd.DataFrame({
        'fecha':          fecha,
        'lat':            lat,          # ← NUEVO
        'lon':            lon,          # ← NUEVO
        'B1': B1.ravel(), 'B2': B2.ravel(), 'B3': B3.ravel(),
        'B4': B4.ravel(), 'B5': B5.ravel(), 'B7': B7.ravel(),
        'NDVI': ndvi.ravel(),
        'NDWI': ndwi.ravel(),
        'EVI':  evi.ravel(),
        'NDBI': ndbi.ravel(),
        'clase_original': clases.ravel(),
    })

    # Binarizar
    df['clase_binaria'] = (df['clase_original'] == urbano).astype(int)

    # Eliminar píxeles sin datos (usando B1 como máscara)
    df = df[df['B1'] > 0].reset_index(drop=True)

    todos.append(df)
    print(f'  {fecha}: {len(df)} píxeles, Urbano={df["clase_binaria"].sum()}')

# Unir todo y guardar
final = pd.concat(todos, ignore_index=True)
final.drop(columns=['clase_original']).to_csv('Dataset.csv', index=False)

print(f'\nCSV generado con {len(final)} píxeles en total')
print(final['clase_binaria'].value_counts())