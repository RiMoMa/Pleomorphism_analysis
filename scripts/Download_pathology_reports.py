path = '/media/ricardo/Datos/Project_Plemorfismo/Pleomorfismo/data/images/DX1_svs'

import os
import re
import requests

# Ruta donde están los archivos .svs

# URL base para la descarga de reportes patológicos
base_url = "https://www.cbioportal.org/patient/pathologyReport?sampleId={}&studyId=brca_tcga_pan_can_atlas_2018"

# Diccionario para almacenar los resultados
download_results = {"success": [], "failed": []}

# Carpeta de destino para los reportes
output_folder = os.path.join(path, "reports")
os.makedirs(output_folder, exist_ok=True)

# Obtener la lista de archivos .svs en la carpeta
svs_filenames = [f for f in os.listdir(path) if f.endswith(".svs")]

# Extraer los identificadores de caso
case_names = []
for filename in svs_filenames:
    match = re.match(r"^([^-]+-[^-]+-[^-]+-[^-]+)", filename)
    if match:
        case_names.append(match.group(1))

# Descargar reportes
for case in case_names:
    sample_id = case[:-3]  # Remover los últimos tres caracteres "-01Z" para obtener el identificador correcto
    print('processing case: {}'.format(sample_id))
    url = base_url.format(sample_id)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Guardar el reporte descargado
            report_path = os.path.join(output_folder, f"{sample_id}.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            download_results["success"].append(sample_id)
        else:
            download_results["failed"].append(sample_id)
    except requests.exceptions.RequestException:
        download_results["failed"].append(sample_id)

# Guardar un archivo con los resultados de la descarga
results_path = os.path.join(output_folder, "download_results.txt")
with open(results_path, "w") as f:
    f.write("Successfully downloaded reports:\n")
    f.write("\n".join(download_results["success"]))
    f.write("\n\nFailed downloads:\n")
    f.write("\n".join(download_results["failed"]))

print(f"Proceso completado. Los resultados se guardaron en {results_path}")
