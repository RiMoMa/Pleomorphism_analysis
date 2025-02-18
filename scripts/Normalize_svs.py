import os
import openslide
from PIL import Image
import cv2
import numpy as np
import staintools
from staintools.tissue_masks.luminosity_threshold_tissue_locator import LuminosityThresholdTissueLocator
from staintools.tissue_masks.luminosity_threshold_tissue_locator import LuminosityThresholdTissueLocator
#from staintools.utils.exceptions import TissueMaskException

from tqdm import tqdm
import subprocess
import json

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

HOVERNET_PYTHON = config["HOVERNET_PYTHON"]
HOVERNET_SCRIPT = config["HOVERNET_SCRIPT"]
HOVERNET_MODEL = config["HOVERNET_MODEL"]
HOVERNET_TYPE_INFO = config["HOVERNET_TYPE_INFO"]

HOVERNET_INPUT_DIR = config["HOVERNET_INPUT_DIR"]
HOVERNET_OUTPUT_DIR = config["HOVERNET_OUTPUT_DIR"]

CUDA_DEVICE = config["CUDA_DEVICE"]
BATCH_SIZE = config["BATCH_SIZE"]
NR_INFERENCE_WORKERS = config["NR_INFERENCE_WORKERS"]
NR_POST_PROC_WORKERS = config["NR_POST_PROC_WORKERS"]
MEM_USAGE = config["MEM_USAGE"]
# ===========================
# FUNCIÓN PARA DETECTAR TEJIDO
# ===========================
def tiene_tejido_staintools(patch, umbral_pixeles=5000):
    """
    Detecta si un parche tiene tejido usando StainTools.
    """

    try:
        mask = LuminosityThresholdTissueLocator.get_tissue_mask(patch)
        count_pixels = np.sum(mask) > umbral_pixeles
    except:
        count_pixels = 0
    return count_pixels  # Al menos 5000 píxeles con tejido

# ===========================
# FUNCIÓN PARA VERIFICAR MAGNIFICACIÓN
# ===========================
def obtener_magnificacion(svs_path):
    slide = openslide.OpenSlide(svs_path)
    try:
        magnificacion = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, "0"))
    except ValueError:
        magnificacion = 0
    return magnificacion

# ===========================
# VERIFICAR MAGNIFICACIÓN DE TODAS LAS IMÁGENES
# ===========================
svs_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".svs")]

if not svs_files:
    print("⚠️ No se encontraron archivos SVS en la carpeta.")
    exit()

# Obtener magnificación de todas las imágenes
magnificaciones = {svs: obtener_magnificacion(os.path.join(INPUT_DIR, svs)) for svs in svs_files}

# Verificar si todas tienen la misma magnificación
valores_magnificacion = set(magnificaciones.values())

if len(valores_magnificacion) > 1:
    print("⚠️ Las imágenes tienen diferentes magnificaciones:")
    for svs, mag in magnificaciones.items():
        print(f"📌 {svs}: {mag}X")
    print("❌ No se puede continuar con el procesamiento.")
    exit()
else:
    magnificacion_unica = list(valores_magnificacion)[0]
    print(f"✅ Todas las imágenes tienen la misma magnificación: {magnificacion_unica}X. Procediendo con el procesamiento.")

# ===========================
# PROCESAR CADA ARCHIVO SVS
# ===========================
for svs_file in svs_files:
    WSI_PATH = os.path.join(INPUT_DIR, svs_file)
    svs_name = os.path.splitext(svs_file)[0]  # Nombre sin extensión

    print(f"📂 Procesando {svs_name}.svs...")

    # Definir rutas de salida
    svs_output_dir = os.path.join(OUTPUT_DIR, svs_name)
    os.makedirs(svs_output_dir, exist_ok=True)
    RECONSTRUCTED_PATH = os.path.join(OUTPUT_DIR, f"{svs_name}.tiff")
    SVS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{svs_name}.svs")

    # Cargar la imagen en un objeto OpenSlide
    slide = openslide.OpenSlide(WSI_PATH)
    width, height = slide.level_dimensions[LEVEL]

    # Cargar imagen de referencia para normalización
    reference_image = np.array(Image.open(REFERENCE_IMAGE_PATH).convert("RGB"))
    vahadane_normalizer = staintools.StainNormalizer(method="vahadane")
    vahadane_normalizer.fit(reference_image)

    print(f"📏 Dimensiones de {svs_name}: {width}x{height}")

    # ===========================
    # EXTRACCIÓN Y NORMALIZACIÓN DE PARCHES (SIN PARALELIZACIÓN)
    # ===========================
    print(f"🖼️ Extrayendo y normalizando parches de {svs_name}...")

    for y in tqdm(range(0, height, PATCH_SIZE), desc=f"📌 {svs_name} - Normalización"):
        for x in range(0, width, PATCH_SIZE):
            patch_filename = os.path.join(svs_output_dir, f"patch_{x}_{y}.png")
            if os.path.exists(patch_filename):
                continue

            patch = slide.read_region((x, y), LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            patch = np.array(patch)
            is_white = np.sum(patch.ravel() < 220) / len(patch.ravel()) > 0.8
            if tiene_tejido_staintools(patch, PATCH_SIZE ** 2 * 0.15) and is_white:
                try:
                    norm_patch = vahadane_normalizer.transform(patch)
                except:
                    norm_patch = patch  # Mantener el parche sin cambios
            else:
                norm_patch = patch  # Mantener fondo sin cambios

            cv2.imwrite(patch_filename, cv2.cvtColor(norm_patch, cv2.COLOR_RGB2BGR))

    print(f"✅ Normalización completada. Parches guardados en {svs_output_dir}")



    # ===========================
    # LLAMAR A HOVERNET PARA SEGMENTACIÓN
    # ===========================
    print(f"🚀 Ejecutando HoverNet para {svs_name} en segundo plano...")




    # Nombre del caso
    input_case_dir = os.path.join(HOVERNET_INPUT_DIR, svs_name)
    output_case_dir = os.path.join(HOVERNET_OUTPUT_DIR, svs_name)

    # Verificar si el script existe
    if not os.path.exists(HOVERNET_SCRIPT):
        print(f"❌ ERROR: No se encontró el script {HOVERNET_SCRIPT}")
        exit()

    # Construir el comando correctamente
    hovernet_command = [
        HOVERNET_PYTHON, HOVERNET_SCRIPT,
        "--gpu=0",
        "--nr_types=6",
        f"--type_info_path={HOVERNET_TYPE_INFO}",
        "--batch_size=1",
        "--model_mode=fast",
        f"--model_path={HOVERNET_MODEL}",
        "--nr_inference_workers=1",
        "--nr_post_proc_workers=1",
        "tile",
        f"--input_dir={input_case_dir}",
        f"--output_dir={output_case_dir}",
        "--mem_usage=0.005",
        "--draw_dot",
        "--save_qupath"
    ]

    # Ejecutar HoverNet
    try:
        hovernet_process = subprocess.Popen(hovernet_command)
        print(f"🚀 HoverNet ejecutándose para {svs_name} en segundo plano...")
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
    except Exception as e:
        print(f"⚠️ Algo salió mal: {e}")

    # ===========================
    # RECONSTRUCCIÓN DE WSI DESDE LOS PARCHES
    # ===========================
    print(f"🔄 Reconstruyendo {svs_name}...")

    reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)
    if not os.path.exists(SVS_OUTPUT_PATH):
        for filename in tqdm(sorted(os.listdir(svs_output_dir)), desc=f"🧩 {svs_name} - Reconstrucción"):
            if filename.endswith(".png"):
                parts = filename.split("_")
                x, y = int(parts[1]), int(parts[2].split(".")[0])

                patch = cv2.imread(os.path.join(svs_output_dir, filename))
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                end_x = x + PATCH_SIZE
                end_y = y + PATCH_SIZE

                if end_x > width:
                    patch = patch[:, :width - x, :]
                if end_y > height:
                    patch = patch[:height - y, :, :]

                reconstructed_image[y:y+patch.shape[0], x:x+patch.shape[1], :] = patch

        Image.fromarray(reconstructed_image).save(RECONSTRUCTED_PATH, format="TIFF", compression="tiff_lzw")
        print(f"✅ Reconstrucción completada. Imagen guardada en {RECONSTRUCTED_PATH}")

        # ===========================
        # CONVERSIÓN DE TIFF A SVS Y ELIMINACIÓN DEL TIFF
        # ===========================
        print(f"🔄 Convirtiendo {svs_name} a SVS...")

        os.system(f"vips tiffsave {RECONSTRUCTED_PATH} {SVS_OUTPUT_PATH} --compression=jpeg --tile --tile-width=512 --tile-height=512 --pyramid")

        if os.path.exists(SVS_OUTPUT_PATH):
            os.remove(RECONSTRUCTED_PATH)
            print(f"✅ {svs_name}.svs generado y TIFF eliminado.")
        else:
            print(f"⚠️ Error en la conversión de {svs_name}. El archivo TIFF no ha sido eliminado.")
    else:
        print('ya existe: ',SVS_OUTPUT_PATH)
print("✅ Todos los SVS han sido procesados correctamente.")
