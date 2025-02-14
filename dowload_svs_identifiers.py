import os
import subprocess

# Ruta del cliente GDC
GDC_CLIENT = "/media/ricardo/Datos/Project_Plemorfismo/Pleomorfismo/APP_TCGA/gdc-client_2.3_Ubuntu_x64/gdc-client"

# Directorios
MANIFESTS_DIR = "data/data_manifests"  # Carpeta donde están los manifests
OUTPUT_DIR = "data/images"  # Carpeta base para descargas

# Crear carpeta de imágenes si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Obtener todos los archivos Manifest_*.txt
manifest_files = [f for f in os.listdir(MANIFESTS_DIR) if f.startswith("Manifest_") and f.endswith(".txt")]

# Iterar sobre cada archivo manifest
for manifest in manifest_files:
    # Extraer el identificador (ejemplo: Manifest_DX1.txt → DX1)
    identifier = manifest.replace("Manifest_", "").replace(".txt", "")

    # Crear carpeta de destino para el identificador
    target_dir = os.path.join(OUTPUT_DIR, identifier)
    os.makedirs(target_dir, exist_ok=True)

    # Ruta completa del archivo manifest
    manifest_path = os.path.join(MANIFESTS_DIR, manifest)

    # Comando para descargar los archivos usando gdc-client
    command = [GDC_CLIENT, "download", "-m", manifest_path, "--dir", target_dir]

    # Ejecutar el comando en la terminal
    print(f"📥 Descargando archivos para {identifier} en {target_dir}...")
    subprocess.run(command, check=True)

    print(f"✅ Descarga completada para {identifier}.\n")

print("🚀 Todas las descargas han finalizado.")
