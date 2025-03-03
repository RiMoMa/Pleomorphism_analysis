import os
import json
import numpy as np
import pandas as pd

def load_config(config_path="configs/config.json"):
    """Carga el archivo de configuración JSON."""
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def load_data(data_path):
    """Carga los datos desde la carpeta procesada."""
    files = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    return files

def load_embeddings(embeddings_path):
    """Carga embeddings preprocesados."""
    embeddings = np.load(os.path.join(embeddings_path, "embeddings.npy"))
    labels = pd.read_csv(os.path.join(embeddings_path, "labels.csv"))
    return embeddings, labels

