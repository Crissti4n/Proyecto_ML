import rasterio
import rasterio.transform
import numpy as np
import pandas as pd
from pyproj import Transformer
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os

# ============================================================
# RUTAS
# ============================================================
ruta_multi  = r'D:/profesores/rriospa/PAE_Cristian/Área multiespectral/'
ruta_modelo = r'D:/profesores/rriospa/PAE_Cristian/Codigos_preparación de datos/modelo_random_forest.pkl'
ruta_csv    = r'D:/profesores/rriospa/PAE_Cristian/Codigos_preparación de datos/Dataset.csv'
ruta_salida = r'D:/profesores/rriospa/PAE_Cristian/Clasificación_RF/'

os.makedirs(ruta_salida, exist_ok=True)

# ============================================================
# 1. CARGAR MODELO
# ============================================================
print('Cargando modelo...')
modelo = joblib.load(ruta_modelo)
print('Modelo cargado.')

# ============================================================
# 2. CARGAR CSV Y MOSTRAR FECHAS DISPONIBLES
# ============================================================
print('\nCargando dataset de entrada...')
df_total = pd.read_csv(ruta_csv)
fechas_disponibles = sorted(df_total['fecha'].astype(str).unique())

print('\n=======================================')
print('  IMÁGENES DISPONIBLES PARA CLASIFICAR')
print('=======================================')
for i, fecha in enumerate(fechas_disponibles):
    n_pixeles = len(df_total[df_total['fecha'].astype(str) == fecha])
    print(f'  [{i+1:2d}] {fecha}   ({n_pixeles:,} píxeles)')
print('=======================================')

# ============================================================
# 3. SELECCIÓN DE FECHA
# ============================================================
while True:
    try:
        seleccion = int(input('\nEscribe el número de la imagen a clasificar: '))
        if 1 <= seleccion <= len(fechas_disponibles):
            fecha_sel = fechas_disponibles[seleccion - 1]
            break
        else:
            print(f'  ⚠️  Escribe un número entre 1 y {len(fechas_disponibles)}')
    except ValueError:
        print('  ⚠️  Entrada inválida, escribe solo el número.')

print(f'\n✅ Fecha seleccionada: {fecha_sel}')

# ============================================================
# 4. FILTRAR PÍXELES DE LA FECHA SELECCIONADA
# ============================================================
df_fecha = df_total[df_total['fecha'].astype(str) == fecha_sel].copy()
print(f'   Píxeles a clasificar: {len(df_fecha):,}')

# Quitar fecha antes de predecir (el modelo no se entrenó con ella)
X_input = df_fecha.drop(columns=['fecha', 'clase_binaria'])

# ============================================================
# 5. CLASIFICAR
# ============================================================
print('\nClasificando...')
predicciones    = modelo.predict(X_input)
probabilidades  = modelo.predict_proba(X_input)[:, 1]
print(f'   Píxeles urbanos detectados:    {predicciones.sum():,}')
print(f'   Píxeles no urbanos detectados: {(predicciones == 0).sum():,}')

# ============================================================
# 6. RECONSTRUIR LA IMAGEN DESDE EL .img ORIGINAL
# ============================================================
ruta_img = ruta_multi + f'A_{fecha_sel}.img'
print(f'\nLeyendo dimensiones desde: {os.path.basename(ruta_img)}')

with rasterio.open(ruta_img) as src:
    filas      = src.height
    cols       = src.width
    transform  = src.transform
    crs        = src.crs
    B1_orig    = src.read(1).astype(float)

print(f'   Dimensiones originales: {filas} filas × {cols} columnas')

# Crear grilla de salida inicializada en -1 (sin dato)
mapa_clase = np.full((filas, cols), -1, dtype=np.int8)
mapa_proba = np.full((filas, cols), np.nan, dtype=np.float32)

# Convertir lat/lon → UTM → fila/col
transformer_inv = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)

lats = df_fecha['lat'].values
lons = df_fecha['lon'].values

x_utm, y_utm = transformer_inv.transform(lons, lats)
filas_px, cols_px = rasterio.transform.rowcol(transform, x_utm, y_utm)
filas_px = np.array(filas_px)
cols_px  = np.array(cols_px)

# Filtrar índices que estén dentro de los límites de la imagen
dentro = (
    (filas_px >= 0) & (filas_px < filas) &
    (cols_px  >= 0) & (cols_px  < cols)
)

mapa_clase[filas_px[dentro], cols_px[dentro]] = predicciones[dentro]
mapa_proba[filas_px[dentro], cols_px[dentro]] = probabilidades[dentro].astype(np.float32)

# ============================================================
# 7. EXPORTAR GEOTIFF DE CLASIFICACIÓN
# ============================================================
ruta_tif = ruta_salida + f'clasificacion_{fecha_sel}.tif'

with rasterio.open(
    ruta_tif, 'w',
    driver    = 'GTiff',
    height    = filas,
    width     = cols,
    count     = 2,
    dtype     = np.float32,
    crs       = crs,
    transform = transform,
    nodata    = -1
) as dst:
    dst.write(mapa_clase.astype(np.float32), 1)
    dst.write(mapa_proba, 2)
    dst.update_tags(1, descripcion='Clase: 0=No urbano, 1=Urbano, -1=Sin dato')
    dst.update_tags(2, descripcion='Probabilidad de ser urbano (0.0 - 1.0)')

print(f'\n✅ GeoTIFF guardado: clasificacion_{fecha_sel}.tif')

# ============================================================
# ============================================================
# 8. RECORTE AUTOMÁTICO (ELIMINAR BORDES NEGROS)
# ============================================================

mask_valid = mapa_clase != -1

filas_validas = np.any(mask_valid, axis=1)
cols_validas  = np.any(mask_valid, axis=0)

r_min, r_max = np.where(filas_validas)[0][[0, -1]]
c_min, c_max = np.where(cols_validas)[0][[0, -1]]

margen = 5
r_min = max(0, r_min - margen)
r_max = min(mapa_clase.shape[0], r_max + margen)
c_min = max(0, c_min - margen)
c_max = min(mapa_clase.shape[1], c_max + margen)


def recortar(arr):
    return arr[r_min:r_max, c_min:c_max]


# Aplicar recorte a mapas
mapa_viz = recortar(mapa_clase.astype(float))
mapa_viz[mapa_viz == -1] = np.nan

mapa_proba_crop = recortar(mapa_proba)


# ============================================================
# 9. VISUALIZACIÓN
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

fig.suptitle(
    f'Clasificación de Cobertura de Suelo — {fecha_sel}',
    fontsize=15,
    fontweight='bold'
)

# ── 9.1 Mapa binario (recortado) ─────────────────────────────
ax1 = axes[0]

cmap_binario = ListedColormap(['#4CAF50', '#E53935'])
im1 = ax1.imshow(
    mapa_viz,
    cmap=cmap_binario,
    vmin=0,
    vmax=1,
    interpolation='nearest'
)

ax1.set_title('Clasificación binaria (recortada)', fontsize=13)
ax1.axis('off')

parche_nu = mpatches.Patch(color='#4CAF50', label='No urbano')
parche_u  = mpatches.Patch(color='#E53935', label='Urbano')

ax1.legend(
    handles=[parche_nu, parche_u],
    loc='lower left',
    fontsize=10,
    framealpha=0.9
)

pct_urbano = predicciones.sum() / len(predicciones) * 100
ax1.set_xlabel(
    f'Urbano: {pct_urbano:.1f}%  |  No urbano: {100-pct_urbano:.1f}%',
    fontsize=10
)

# ── 9.2 Mapa de probabilidad (recortado) ────────────────────
ax2 = axes[1]

im2 = ax2.imshow(
    mapa_proba_crop,
    cmap='RdYlGn_r',
    vmin=0,
    vmax=1,
    interpolation='nearest'
)

ax2.set_title('Probabilidad de ser Urbano (recortada)', fontsize=13)
ax2.axis('off')

plt.colorbar(
    im2,
    ax=ax2,
    fraction=0.046,
    pad=0.04,
    label='Probabilidad (0 = No urbano, 1 = Urbano)'
)

plt.tight_layout()

ruta_png = ruta_salida + f'mapa_{fecha_sel}.png'
plt.savefig(ruta_png, dpi=200, bbox_inches='tight')
plt.show()

print(f'✅ Mapa guardado: {ruta_png}')
print('\n--- Proceso completado ---')