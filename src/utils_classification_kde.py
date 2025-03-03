import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('TkAgg')   # O usa 'QtAgg' si tienes una versión más reciente
import matplotlib.pyplot as plt


# 1️⃣ **Cargar Embeddings desde .npy y Procesar Pacientes**
def load_dataset_embeddings(bracs_path,Atypia_path,TCGA_path):
    # Cargar embeddings desde archivos .npy
    bracs_embeddings = np.load(bracs_path, allow_pickle=True)
    atypia_embeddings = np.load(Atypia_path, allow_pickle=True)
    tcga_embeddings = np.load(TCGA_path, allow_pickle=True)

    return bracs_embeddings, atypia_embeddings, tcga_embeddings


# 2️⃣ **Procesar Casos de TCGA y Extraer Información**
def process_tcga_cases(tcga_embeddings, tcga_files):
    df = pd.DataFrame(tcga_files, columns=["archivo"])
    df['embeddings'] = list(tcga_embeddings)
    df["caso"] = df["archivo"].apply(lambda x: x.split('.')[0])
    df_sorted = df.sort_values(by="caso").reset_index(drop=True)
    return df_sorted


# 3️⃣ **Aplicar KDE Global**
def apply_global_kde(predictions):
    predictions_array = np.column_stack(predictions)
    kde_global = gaussian_kde(predictions_array.T)
    x_grid, y_grid = np.meshgrid(np.linspace(predictions_array[:, 0].min(), predictions_array[:, 0].max(), 100),
                                 np.linspace(predictions_array[:, 1].min(), predictions_array[:, 1].max(), 100))
    z_global = kde_global(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(x_grid, y_grid, z_global, levels=20, cmap="viridis")
    plt.colorbar(label="Densidad KDE Global")
    plt.scatter(predictions_array[:, 0], predictions_array[:, 1], s=5, color="red", alpha=0.3)
    plt.xlabel("Predicción de Tipo de Tumor")
    plt.ylabel("Predicción de Pleomorfismo")
    plt.title("Densidad KDE Global de Predicciones")
    plt.show()

import os
# 4️⃣ **Aplicar KDE por Paciente y Medir Variabilidad**
def compute_patient_kde_variability(predictions, tcga_df, output_dir="kde_per_patient"):
    os.makedirs(output_dir, exist_ok=True)
    patient_kde_variability = {}
    predictions_array = np.column_stack(predictions)
    tcga_df["tumor_prediction"] = predictions_array[:, 0]
    tcga_df["pleomorphism_prediction"] = predictions_array[:, 1]

    for patient_id, group in tcga_df.groupby("caso"):
        kde_patient = gaussian_kde(np.vstack([group["tumor_prediction"], group["pleomorphism_prediction"]]))
        densities = kde_patient(np.vstack([group["tumor_prediction"], group["pleomorphism_prediction"]]))
        kde_variability = np.std(densities)
        patient_kde_variability[patient_id] = kde_variability

        # Generar y guardar la gráfica de KDE por paciente
        x_grid, y_grid = np.meshgrid(np.linspace(group["tumor_prediction"].min(), group["tumor_prediction"].max(), 100),
                                     np.linspace(group["pleomorphism_prediction"].min(),
                                                 group["pleomorphism_prediction"].max(), 100))
        z_patient = kde_patient(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(x_grid, y_grid, z_patient, levels=20, cmap="viridis")
        plt.colorbar(label="Densidad KDE Paciente")
        plt.scatter(group["tumor_prediction"], group["pleomorphism_prediction"], s=5, color="red", alpha=0.3)
        plt.xlabel("Predicción de Tipo de Tumor")
        plt.ylabel("Predicción de Pleomorfismo")
        plt.title(f"Densidad KDE para Paciente {patient_id}")
        plt.savefig(os.path.join(output_dir, f"kde_patient_{patient_id}.png"))
        plt.close()

    return pd.DataFrame.from_dict(patient_kde_variability, orient="index", columns=["kde_variability"])


# 5️⃣ **Entrenar Modelos de Clasificación para Pleomorfismo y Tipo de Cáncer**
def train_classification_models(embeddings, labels, model_type="tumor_type"):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Modelo de clasificación para {model_type}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

