import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from src.utils_classification_kde import *
def load_embeddings_dataset(path_data_embeddings):
    """
    Carga los embeddings, etiquetas y nombres desde un archivo .npy.

    Args:
        path_data_embeddings (str): Ruta al archivo .npy que contiene los embeddings.

    Returns:
        tuple: (embeddings, labels, names), donde:
            - embeddings (np.ndarray): Matriz de embeddings.
            - labels (list): Lista de etiquetas correspondientes.
            - names (list): Lista de nombres asociados a los datos.
    """
    try:
        data = np.load(path_data_embeddings, allow_pickle=True).item()  # Cargar como diccionario
        embeddings = data.get("embeddings", np.array([]))
        labels = data.get("labels", [])
        names = data.get("names", [])

        return embeddings, labels, names
    except Exception as e:
        print(f"Error al cargar embeddings desde {path_data_embeddings}: {e}")
        return None, None, None

def save_embeddings(path, embeddings, labels, names):
    """
    Guarda los embeddings, etiquetas y nombres en un archivo .npy.

    Args:
        path (str): Ruta del archivo de salida.
        embeddings (np.ndarray): Matriz de embeddings.
        labels (list): Lista de etiquetas.
        names (list): Lista de nombres.
    """
    data = {
        "embeddings": embeddings,
        "labels": labels,
        "names": names
    }
    np.save(path, data)


import os
import json
import mlflow
from src.data_loader import load_data, load_embeddings
from src.embeddings_extractor import extract_and_save_embeddings
#from src.nuclei_segmentation import segment_nuclei
#from src.clustering import cluster_embeddings, plot_clusters
#from src.training import train_classifiers
#from src.survival_analysis import survival_analysis
#from src.evaluation import evaluate_models
from src.train_eval_embeddings_classifiers import train_classifiers
from src.train_eval_embeddings_classifiers import train_and_evaluate_classifiers
from src.train_eval_embeddings_classifiers import train_and_evaluate_classifiers_grouped
from src.train_eval_embeddings_classifiers import predict_without_labels
import matplotlib
matplotlib.use('TkAgg')   # O usa 'QtAgg' si tienes una versión más reciente
import matplotlib.pyplot as plt

# Cargar configuración
def load_config(config_path="configs/config.json"):
    with open(config_path, 'r') as file:
        return json.load(file)


def load_survival_data(survival_json):
    # Cargar y procesar datos de supervivencia desde JSON
    with open(survival_json, "r") as f:
        survival_json = json.load(f)

    survival_data = []
    for donor in survival_json[0]["donors"]:
        survival_data.append({
            "caso": donor["submitter_id"],
            "time": donor["time"],
            "status": 0 if donor["censored"] else 1  # 1 si murió, 0 si censurado
        })

    survival_df = pd.DataFrame(survival_data)
    return survival_df

config = load_config()
mlflow.set_tracking_uri(config["mlflow_tracking_uri"])

data_path = config["data_path"]
embeddings_dir = config["embeddings_path"]
datasets_dir = config["datasets_dirs"]
survival_json = config["survival_json"]

def map_labels(labels,class_mapping):
    """Convierte las etiquetas originales en las categorías Normal, Benign y Malignant."""
    return np.array([class_mapping[label] for label in labels])

def extract_embeddings(dataset, embeddings_dir):
    print(f"Running {dataset}")
        #data = load_data(os.path.join(data_path, dataset))
    data = os.path.join(data_path, dataset)
    embeddings_output_path = os.path.join(embeddings_dir, dataset)
    # 2. Extraer embeddings
    print(f"Extracting embeddings for {dataset}")

    path_data_embeddings = os.path.join(embeddings_dir, dataset+'.npy')
    if not os.path.exists(path_data_embeddings):
        embeddings, labels,names = extract_and_save_embeddings(data, embeddings_output_path , model_name="hf-hub:MahmoodLab/UNI2-h")
        save_embeddings(path_data_embeddings, embeddings, labels, names)

    else:
        embeddings, labels,names = load_embeddings_dataset(path_data_embeddings)
    return embeddings, labels, names








            #         print("here I am")
            #
            # predictions, probabilities = predict_without_labels(all_embeddings, all_labels, TCGA_embeddings, model_type="logistic_regression")
            #
            # # Crear DataFrame
            # df = pd.DataFrame(TCGA_names, columns=["archivo"])
            # df['predictions'] = predictions
            # df['probabilities'] = probabilities
            # df['embeddings'] = TCGA_embeddings
def kde_analysis(all_data):
    tcga_df = process_tcga_cases(all_data['sample_tcga_embeddings'],all_data['sample_tcga_names'])
    model_tumor = train_classification_models(np.vstack(all_data['Bracs_embeddings']), all_data['Bracs_labels'],
                                                      "tumor_type")
    model_pleomorphism = train_classification_models(np.vstack(all_data['ATYPIA_classes_embeddings']),
                                                     all_data['ATYPIA_classes_labels'],
                                                     "pleomorphism")


    tumor_predictions = model_tumor.predict(all_data['sample_tcga_embeddings'])
    pleomorphims_prediction = model_pleomorphism.predict(all_data['sample_tcga_embeddings'])

    # Definir el mapeo de clases
    class_mapping = {
        '0_N': 0, '1_PB': 0, '2_UDH': 0,  # Normal
        '3_FEA': 1, '4_ADH': 1,  # Benign
        '5_DCIS': 2, '6_IC': 2,  # Malignant
        'clase_1': 3,
        'clase_2': 4,
        'clase_3': 5
    }

    tumor_predictions_mapped =  map_labels(tumor_predictions,class_mapping   )
    pleomorphims_predictions_mapped = map_labels(pleomorphims_prediction,class_mapping)

    # Aplicar KDE Global a BRACS y ATYPIA
    apply_global_kde([tumor_predictions_mapped, pleomorphims_predictions_mapped])

    # Medir variabilidad intra-paciente en TCGA
    patient_kde_variability_df = compute_patient_kde_variability([tumor_predictions_mapped, pleomorphims_predictions_mapped], tcga_df)



#survival analisis


    print(patient_kde_variability_df.head())
    return patient_kde_variability_df



        #
        #
        #     exit()
        #     # 3. Segmentación de núcleos
        #     print("Segmentando núcleos...")
        #     segment_nuclei(config["data_path"], config["nuclei_path"])
        #
        #     # 4. Clustering y visualización
        #     print("Aplicando clustering...")
        #     reduced_embeddings, cluster_labels = cluster_embeddings(all_embeddings)
        #     plot_clusters(reduced_embeddings, cluster_labels)
        #
        #     # 5. Entrenamiento de clasificadores
        #     print("Entrenando modelos de clasificación...")
        #     train_classifiers(embeddings, labels, config["classification_models"])
        #
        #     # 6. Análisis de sobrevida
        #     print("Realizando análisis de sobrevida...")
        #     survival_analysis(config["data_path"])
        #
        #     # 7. Evaluación de modelos
        #     print("Evaluando modelos...")
        #     evaluate_models(config["model_checkpoint"])
        #
        #     mlflow.log_artifact("results/")
        #     print("Experimento completado exitosamente.")
        # except Exception as e:
        #     print(f"Error en el experimento: {e}")
        #     mlflow.log_param("error", str(e))

def survival_analysis(patient_kde_variability_df, survival_data):
    survival_df = survival_data.merge(patient_kde_variability_df, left_on="caso", right_index=True)

    # Ajustar modelo de Cox
    cph = CoxPHFitter()
    cph.fit(survival_df, duration_col="time", event_col="status")
    cph.print_summary()
    cph.plot()
    plt.savefig("cox_regression_plot.png")
    plt.close()

    # Ajustar curvas de Kaplan-Meier
    kmf = KaplanMeierFitter()

    high_variability = survival_df[survival_df["kde_variability"] >= survival_df["kde_variability"].median()]
    low_variability = survival_df[survival_df["kde_variability"] < survival_df["kde_variability"].median()]

    plt.figure(figsize=(8, 6))
    kmf.fit(high_variability["time"], high_variability["status"], label="Alta Variabilidad KDE")
    kmf.plot_survival_function()

    kmf.fit(low_variability["time"], low_variability["status"], label="Baja Variabilidad KDE")
    kmf.plot_survival_function()

    plt.title("Curvas de Supervivencia Kaplan-Meier")
    plt.xlabel("Tiempo de Supervivencia")
    plt.ylabel("Probabilidad de Supervivencia")
    plt.legend()
    plt.savefig("kaplan_meier_plot.png")
    plt.close()

if __name__ == "__main__":
    all_data = {}
    all_results={}
    for dataset in datasets_dir:
        print(f"Running {dataset}")
        dataset_embeddings, dataset_labels, dataset_names = extract_embeddings(dataset, embeddings_dir)
        print("Classification in a 5-fold validation for ",dataset)
       # results = train_classifiers(dataset_embeddings, dataset_labels, config["classification_models"])
        #all_results[f"{dataset}_results"] = results
        all_data[f"{dataset}_embeddings"] = dataset_embeddings
        all_data[f"{dataset}_labels"] = dataset_labels
        all_data[f"{dataset}_names"] = dataset_names
    print("Evaluating Bracs using Train and Test Sets"
          )
    # results_train_test = train_and_evaluate_classifiers(all_data["Bracs_embeddings"], all_data["Bracs_labels"]
    #                                                     ,all_data["Bracs_test_embeddings"],
    #                                                     all_data["Bracs_test_labels"],
    #                                                     config["classification_models"])

    print("evaluating Gruping in Bracs train and test sets")
    #
    # results_train_test_grouped = train_and_evaluate_classifiers_grouped(all_data["Bracs_embeddings"], all_data["Bracs_labels"]
    #                                                     ,all_data["Bracs_test_embeddings"],
    #                                                     all_data["Bracs_test_labels"],
    #                                                     config["classification_models"])

    patient_kde_variability_df = kde_analysis(all_data)

    #SURVIVAL ANALYSIS

    survival_df = load_survival_data(survival_json)

    survival_analysis(patient_kde_variability_df, survival_df )


    # Acceder a los datos
    print(all_data["dataset1_embeddings"])  # Forma segura de acceder a los datos
