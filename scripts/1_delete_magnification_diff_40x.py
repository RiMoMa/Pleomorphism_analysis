import os
import openslide

# Ruta de la carpeta con los archivos SVS
INPUT_DIR = "/media/ricardo/Datos/Project_Plemorfismo/Pleomorfismo/data/images/DX1_svs"

# Función para obtener la magnificación de un archivo SVS
def obtener_magnificacion(svs_path):
    slide = openslide.OpenSlide(svs_path)
    try:
        magnificacion = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, "0"))
    except ValueError:
        magnificacion = 0
    return magnificacion

# Obtener la lista de archivos SVS en la carpeta
svs_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".svs")]

# Revisar y eliminar archivos con magnificación diferente a 40X
for svs_file in svs_files:
    svs_path = os.path.join(INPUT_DIR, svs_file)
    magnificacion = obtener_magnificacion(svs_path)

    if magnificacion != 40:
        print(f"🗑️ Eliminando {svs_file} (Magnificación: {magnificacion}X)")
        os.remove(svs_path)

print("✅ Eliminación completada. Solo quedan archivos de 40X.")
