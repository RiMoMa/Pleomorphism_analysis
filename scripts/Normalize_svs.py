import os
import cv2
import numpy as np
import openslide
from PIL import Image

from tqdm import tqdm
from PIL import Image
import staintools
# ===========================
# CONFIGURACIÓN GENERAL
# ===========================
WSI_PATH = "/media/ricardo/Datos/Project_Plemorfismo/Pleomorfismo/data/images/HE/0a5410d7-0f5c-4dda-986e-d857d176498f/HCM-CSHL-0366-C50-01A-01-S1-HE.30E0E448-BC32-4FCA-99F8-CF5E8C283352.svs"
OUTPUT_DIR = "salida_parches/"
PATCH_SIZE = 512*4  # Ajustar según la memoria disponible
LEVEL = 0  # Nivel de resolución del WSI

# ===========================
# NORMALIZACIÓN DE COLOR (Vahadane) Y EXTRACCIÓN DE PARCHES
# ===========================
print("Cargando WSI y preparando normalización...")

# Cargar la imagen en un objeto OpenSlide
slide = openslide.OpenSlide(WSI_PATH)
width, height = slide.level_dimensions[LEVEL]

# Cargar imagen de referencia para normalización
REFERENCE_IMAGE = np.array(Image.open('/media/ricardo/Datos/Project_Plemorfismo/Pleomorfismo/testA_35.bmp').convert("RGB"))
vahadane_normalizer = staintools.StainNormalizer(method="vahadane")
vahadane_normalizer.fit(REFERENCE_IMAGE)

# Crear carpeta de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Procesando parches...")

# Iterar por toda la imagen en parches
for y in tqdm(range(0, height, PATCH_SIZE), desc="Filas procesadas"):
    for x in range(0, width, PATCH_SIZE):
        # Leer parche
        patch = slide.read_region((x, y), LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
        patch = np.array(patch)
        # Aplicar normalización de color si el parche no está vacío
        # Aplicar normalización de color si el parche no está vacío
        is_white = np.sum(patch.ravel() > 220) / len(patch.ravel()) > 0.9

        if is_white:  # Evitar parches vacíos
            try:
                norm_patch = vahadane_normalizer.transform(patch)
            except staintools.miscellaneous.exceptions.TissueMaskException:
                print(f"⚠️ Advertencia: Parche vacío detectado en ({x}, {y}), omitiendo...")
                norm_patch = patch  # Mantener el parche sin cambios
        else:
            norm_patch = patch  # Mantener fondo sin cambios

        # Guardar parche normalizado
        patch_filename = os.path.join(OUTPUT_DIR, f"patch_{x}_{y}.png")
        cv2.imwrite(patch_filename, cv2.cvtColor(norm_patch, cv2.COLOR_RGB2BGR))

print(f"Normalización completada. Parches guardados en {OUTPUT_DIR}")

# ===========================
# RECONSTRUCCIÓN DEL WSI DESDE LOS PARCHES
# ===========================
RECONSTRUCTED_PATH = "imagen_reconstruida.tiff"

# Crear una imagen vacía para la reconstrucción
reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)

print("Reconstruyendo la imagen...")

# Cargar los parches y ensamblarlos en la imagen
for filename in tqdm(sorted(os.listdir(OUTPUT_DIR)), desc="Parches ensamblados"):
    if filename.endswith(".png"):
        # Extraer coordenadas del nombre del archivo (ejemplo: patch_1024_512.png)
        parts = filename.split("_")
        x, y = int(parts[1]), int(parts[2].split(".")[0])

        # Leer el parche
        patch = cv2.imread(os.path.join(OUTPUT_DIR, filename))
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)  # Convertir de OpenCV a formato RGB

        # Insertar el parche en la imagen reconstruida
        reconstructed_image[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :] = patch

# Guardar la imagen reconstruida en formato TIFF
Image.fromarray(reconstructed_image).save(RECONSTRUCTED_PATH, format="TIFF")

print(f"Reconstrucción completada. Imagen guardada en {RECONSTRUCTED_PATH}")
