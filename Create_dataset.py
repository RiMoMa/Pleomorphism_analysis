import json
import os
import pandas as pd

# Definir rutas
json_file_path = "overall-survival-plot.2025-02-14.json"
manifest_file_path = "gdc_manifest.2025-02-14.104532.txt"
output_folder = "data/data_manifests"

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# 1️⃣ Cargar el JSON y extraer submitter_id
with open(json_file_path, "r") as file:
    data = json.load(file)

submitter_ids = [donor["submitter_id"] for donor in data[0]["donors"]]
print(f"Submitter IDs extraídos: {submitter_ids}")

# 2️⃣ Cargar el archivo Manifest.txt
df = pd.read_csv(manifest_file_path, sep="\t")

# 3️⃣ Filtrar filas donde 'filename' comience con algún submitter_id
filtered_df = df[df["filename"].str.startswith(tuple(submitter_ids))]

# Guardar el manifiesto filtrado general
filtered_manifest_path = os.path.join(output_folder, "Filtered_Manifest.txt")
filtered_df.to_csv(filtered_manifest_path, sep="\t", index=False)
print(f"Archivo filtrado guardado en: {filtered_manifest_path}")

# 4️⃣ Extraer identificador después del último '-' y antes del primer '.'
filtered_df["extracted_id"] = filtered_df["filename"].str.extract(r'-(\w+)\.')

# 5️⃣ Obtener valores únicos y generar manifiestos por identificador
unique_identifiers = filtered_df["extracted_id"].dropna().unique()
for identifier in unique_identifiers:
    subset_df = filtered_df[filtered_df["extracted_id"] == identifier].drop(columns=["extracted_id"])  # Eliminar la columna extra
    file_name = f"Manifest_{identifier}.txt"
    file_path = os.path.join(output_folder, file_name)
    subset_df.to_csv(file_path, sep="\t", index=False)
    print(f"Archivo creado: {file_path} con {len(subset_df)} registros.")

# 6️⃣ Guardar la lista de identificadores únicos
# unique_ids_path = os.path.join(output_folder, "unique_identifiers.txt")
# unique_df = pd.DataFrame(unique_identifiers, columns=["unique_identifier"])
# unique_df.to_csv(unique_ids_path, index=False, sep="\t")
# print(f"Lista de identificadores únicos guardada en: {unique_ids_path}")

print("✅ Proceso completado.")
