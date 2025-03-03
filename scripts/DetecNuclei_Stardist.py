import os
import json
import openslide
from PIL import Image
import time
import cv2
import numpy as np
import subprocess
from tqdm import tqdm
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from skimage import measure
# ===========================
# CARGAR CONFIGURACIÓN DESDE JSON
# ===========================
config_path = "config.json"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"❌ No se encontró {config_path}. Asegúrate de definirlo correctamente.")

with open(config_path, "r") as f:
    config = json.load(f)

# Asignar variables desde JSON
INPUT_DIR = config["INPUT_DIR"]
OUTPUT_DIR = config["OUTPUT_DIR"]
PATCH_SIZE = config["PATCH_SIZE"]
LEVEL = config["LEVEL"]
REFERENCE_IMAGE_PATH = config["REFERENCE_IMAGE_PATH"]

ENABLE_NORMALIZATION = config["ENABLE_NORMALIZATION"]
ENABLE_NUCLEI_DETECTION = config["ENABLE_NUCLEI_DETECTION"]


HOVERNET_INPUT_DIR = config["HOVERNET_INPUT_DIR"]
HOVERNET_OUTPUT_DIR = config["HOVERNET_OUTPUT_DIR"]

# ===========================
# FUNCIÓN PARA DETECTAR TEJIDO
# ===========================


def detect_histological_tissue(img, intensity_threshold=200, var_threshold=500, saturation_threshold=20):
    """
    Detecta si un parche tiene tejido histológico basado en:
    1. Intensidad media en escala de grises.
    2. Varianza de píxeles.
    3. Porcentaje de píxeles con saturación significativa en el espacio HSV.
    """
    # Cargar imagen

    if img is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return False

    # Convertir a escala de grises y calcular intensidad media
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)

    # Calcular varianza
    variance = np.var(gray)

    # Convertir a HSV y analizar saturación
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    percentage_saturated = np.mean(saturation > saturation_threshold) * 100  # Porcentaje de píxeles con color

    # Criterios de detección de tejido
    is_tissue = (mean_intensity < intensity_threshold) and (variance > var_threshold) and (percentage_saturated > 5)

    return is_tissue
def tiene_tejido_staintools(patch, umbral_pixeles=5000):
    try:
        mask = LuminosityThresholdTissueLocator.get_tissue_mask(patch)
        return np.sum(mask) > umbral_pixeles
    except:
        return False

# ===========================
# FUNCIÓN PARA VERIFICAR MAGNIFICACIÓN
# ===========================
def obtener_magnificacion(svs_path):
    slide = openslide.OpenSlide(svs_path)
    try:
        return float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, "0"))
    except ValueError:
        return 0

# ===========================
# VERIFICAR MAGNIFICACIÓN DE TODAS LAS IMÁGENES
# ===========================
svs_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".svs")]

if not svs_files:
    print("⚠️ No se encontraron archivos SVS en la carpeta.")
    exit()

magnificaciones = {svs: obtener_magnificacion(os.path.join(INPUT_DIR, svs)) for svs in svs_files}
valores_magnificacion = set(magnificaciones.values())

if len(valores_magnificacion) > 1:
    print("⚠️ Las imágenes tienen diferentes magnificaciones:")
    for svs, mag in magnificaciones.items():
        print(f"📌 {svs}: {mag}X")
    print("❌ No se puede continuar con el procesamiento.")
    exit()
else:
    print(f"✅ Todas las imágenes tienen la misma magnificación: {list(valores_magnificacion)[0]}X.")

# ===========================
# PROCESAR CADA ARCHIVO SVS
# ===========================
model = StarDist2D.from_pretrained('2D_versatile_he')

for svs_file in svs_files:
    WSI_PATH = os.path.join(INPUT_DIR, svs_file)
    svs_name = os.path.splitext(svs_file)[0]

    print(f"📂 Procesando {svs_name}.svs...")

    svs_output_dir = os.path.join(OUTPUT_DIR, svs_name)
    os.makedirs(svs_output_dir, exist_ok=True)
    RECONSTRUCTED_PATH = os.path.join(svs_output_dir, f"{svs_name}.tiff")
    SVS_OUTPUT_PATH = os.path.join(svs_output_dir, f"{svs_name}.svs")

    slide = openslide.OpenSlide(WSI_PATH)
    width, height = slide.level_dimensions[LEVEL]

    # ===========================
    # NORMALIZACIÓN DE COLOR (Opcional)
    # ===========================
    if ENABLE_NORMALIZATION:
        import staintools
        from staintools.tissue_masks.luminosity_threshold_tissue_locator import LuminosityThresholdTissueLocator

        print("🖼️ Normalización habilitada...")
        reference_image = np.array(Image.open(REFERENCE_IMAGE_PATH).convert("RGB"))
        vahadane_normalizer = staintools.StainNormalizer(method="vahadane")
        vahadane_normalizer.fit(reference_image)
    else:
        print("🛑 Normalización deshabilitada...")
    #1539 × 1376
    for y in tqdm(range(0, height, 1376), desc=f"📌 {svs_name} - Procesamiento"):
        for x in range(0, width, 1539):
            patch_filename = os.path.join(svs_output_dir, f"patch_{x}_{y}.png")


            if os.path.exists(patch_filename):
                continue
            patch = slide.read_region((x, y), LEVEL, (1539, 1376)).convert("RGB")
            patch = np.array(patch)

            if not detect_histological_tissue(patch):
                continue

            if ENABLE_NORMALIZATION :
                try:
                    norm_patch = vahadane_normalizer.transform(patch)
                except:
                    norm_patch = patch
            else:
                norm_patch = patch

            cv2.imwrite(patch_filename, cv2.cvtColor(norm_patch, cv2.COLOR_RGB2BGR))

            if ENABLE_NUCLEI_DETECTION:
                input_case_dir = os.path.join(HOVERNET_INPUT_DIR, svs_name)
                output_case_dir = os.path.join(HOVERNET_OUTPUT_DIR, svs_name)
                os.makedirs(output_case_dir, exist_ok=True)
                output_path = os.path.join(output_case_dir, f'patch_{x}_{y}_inst_map.png')
                if os.path.exists(output_path):
                    continue

             #   print(f"🚀 Ejecutando Startdist para {svs_name}...")
                labels, _ = model.predict_instances(normalize(norm_patch))
                time.sleep(0.5)  # Espera 0.5 segundos
                np.save(os.path.join(output_case_dir, f'patch_{x}_{y}_inst_map.npy'), labels)
                #plt.imsave(os.path.join(output_case_dir, f'{os.path.splitext(svs_name)[0]}_inst_map.png'), labels,
                #           cmap='jet')

                # Find contours for visualization
                contours = measure.find_contours(labels, level=0.5)

                # Overlay contours on the input image
                fig, ax = plt.subplots()
                ax.imshow(norm_patch, cmap='gray')
                ax.axis('off')

                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], linewidth=0.4, color='#52f212')

                # Save the visualization
                fig.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
                plt.close(fig)


            else:
                print("🛑 Detección de núcleos deshabilitada...")

    print(f"✅ Parches guardados en {svs_output_dir}")

    # ===========================
    # LLAMAR A HOVERNET PARA DETECCIÓN DE NÚCLEOS (Opcional)
    # ===========================





print("✅ Todos los SVS han sido procesados correctamente.")
