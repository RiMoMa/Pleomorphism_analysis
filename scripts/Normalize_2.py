import os
import json
import openslide
from PIL import Image
import cv2
import numpy as np
import staintools
import subprocess
from tqdm import tqdm
from staintools.tissue_masks.luminosity_threshold_tissue_locator import LuminosityThresholdTissueLocator

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
        print("🖼️ Normalización habilitada...")
        reference_image = np.array(Image.open(REFERENCE_IMAGE_PATH).convert("RGB"))
        vahadane_normalizer = staintools.StainNormalizer(method="vahadane")
        vahadane_normalizer.fit(reference_image)
    else:
        print("🛑 Normalización deshabilitada...")

    for y in tqdm(range(0, height, PATCH_SIZE), desc=f"📌 {svs_name} - Procesamiento"):
        for x in range(0, width, PATCH_SIZE):
            patch_filename = os.path.join(svs_output_dir, f"patch_{x}_{y}.png")
            if os.path.exists(patch_filename):
                continue

            patch = slide.read_region((x, y), LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            patch = np.array(patch)

            if ENABLE_NORMALIZATION and tiene_tejido_staintools(patch, PATCH_SIZE ** 2 * 0.15):
                try:
                    norm_patch = vahadane_normalizer.transform(patch)
                except:
                    norm_patch = patch
            else:
                norm_patch = patch

            cv2.imwrite(patch_filename, cv2.cvtColor(norm_patch, cv2.COLOR_RGB2BGR))

    print(f"✅ Parches guardados en {svs_output_dir}")

    # ===========================
    # LLAMAR A HOVERNET PARA DETECCIÓN DE NÚCLEOS (Opcional)
    # ===========================



    if ENABLE_NUCLEI_DETECTION:
        print(f"🚀 Ejecutando HoverNet para {svs_name}...")

        input_case_dir = os.path.join(HOVERNET_INPUT_DIR, svs_name)
        output_case_dir = os.path.join(HOVERNET_OUTPUT_DIR, svs_name)

        hovernet_command = [
            HOVERNET_PYTHON, HOVERNET_SCRIPT,
            f"--gpu={CUDA_DEVICE}",
            "--nr_types=6",
            f"--type_info_path={HOVERNET_TYPE_INFO}",
            f"--batch_size={BATCH_SIZE}",
            "--model_mode=fast",
            f"--model_path={HOVERNET_MODEL}",
            f"--nr_inference_workers={NR_INFERENCE_WORKERS}",
            f"--nr_post_proc_workers={NR_POST_PROC_WORKERS}",
            "tile",
            f"--input_dir={input_case_dir}",
            f"--output_dir={output_case_dir}",
            f"--mem_usage={MEM_USAGE}",
            "--draw_dot",
            "--save_qupath"
        ]

        # Ejecutar HoverNet y esperar a que termine
        result = subprocess.run(hovernet_command, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ HoverNet finalizado correctamente para {svs_name}.")
        else:
            print(f"❌ Error en HoverNet para {svs_name}. Código de salida: {result.returncode}")
            print(result.stderr)

    else:
        print("🛑 Detección de núcleos deshabilitada...")

print("✅ Todos los SVS han sido procesados correctamente.")
