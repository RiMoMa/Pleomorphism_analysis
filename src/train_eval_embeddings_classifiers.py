
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # O usa 'QtAgg' si tienes una versión más reciente
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def train_classifiers(embeddings, labels, model_types):
    print("Entrenando clasificadores...")
    unique_labels = np.unique(labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_index[label] for label in labels])

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}

    for model_type in model_types:
        accuracy_list, precision_list, recall_list, f1_list = [], [], [], []
        conf_matrix_accumulated = np.zeros((len(unique_labels), len(unique_labels)))

        for train_idx, test_idx in skf.split(embeddings, y):
            X_train, X_test = embeddings[train_idx], embeddings[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if model_type == "logistic_regression":
                model = LogisticRegression(max_iter=1000)
            elif model_type == "knn":
                model = KNeighborsClassifier(n_neighbors=100)
            elif model_type == "xgboost":
                model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            else:
                raise ValueError(f"Modelo {model_type} no soportado.")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy_list.append(accuracy_score(y_test, y_pred))
            precision_list.append(precision_score(y_test, y_pred, average='macro'))
            recall_list.append(recall_score(y_test, y_pred, average='macro'))
            f1_list.append(f1_score(y_test, y_pred, average='macro'))
            conf_matrix_accumulated += confusion_matrix(y_test, y_pred, labels=range(len(unique_labels)))

        results[model_type] = {
            "accuracy": np.mean(accuracy_list),
            "precision": np.mean(precision_list),
            "recall": np.mean(recall_list),
            "f1_score": np.mean(f1_list),
            "conf_matrix": conf_matrix_accumulated
        }

    print("Resultados de clasificación:")
    for model_type, metrics in results.items():
        print(
            f"{model_type} -> Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")

    # Visualización de la Matriz de Confusión Acumulada
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_accumulated, annot=True, fmt=".0f", cmap="Blues", xticklabels=unique_labels,
                yticklabels=unique_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Matriz de Confusión Acumulada")
    plt.show()

    return results


def train_and_evaluate_classifiers(embeddings_train, labels_train, embeddings_test, labels_test, model_types):
    print("Entrenando clasificadores...")

    unique_labels = np.unique(labels_train)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    y_train = np.array([label_to_index[label] for label in labels_train])
    y_test = np.array([label_to_index[label] for label in labels_test])

    results = {}

    for model_type in model_types:
        if model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "knn":
            model = KNeighborsClassifier(n_neighbors=100)
        elif model_type == "xgboost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        else:
            raise ValueError(f"Modelo {model_type} no soportado.")

        model.fit(embeddings_train, y_train)
        y_pred = model.predict(embeddings_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_test, y_pred, labels=range(len(unique_labels)))

        results[model_type] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "conf_matrix": conf_matrix
        }

        print(
            f"{model_type} -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # Visualización de la Matriz de Confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt=".0f", cmap="Blues", xticklabels=unique_labels,
                    yticklabels=unique_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Matriz de Confusión - {model_type}")
        plt.show()

    return results



# Definir el mapeo de clases
class_mapping = {
    '0_N': 'Normal', '1_PB': 'Normal', '2_UDH': 'Normal',
    '3_FEA': 'Benign', '4_ADH': 'Benign',
    '5_DCIS': 'Malignant', '6_IC': 'Malignant',

}

def map_labels(labels):
    """Convierte las etiquetas originales en las categorías Normal, Benign y Malignant."""
    return np.array([class_mapping[label] for label in labels])

def train_and_evaluate_classifiers_grouped(embeddings_train, labels_train, embeddings_test, labels_test, model_types):
    print("Entrenando clasificadores con clases agrupadas...")

    # Reasignar etiquetas a las nuevas categorías
    y_train = map_labels(labels_train)
    y_test = map_labels(labels_test)

    unique_labels = np.unique(y_train)  # Obtener las clases únicas después de la agrupación
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    y_train = np.array([label_to_index[label] for label in y_train])
    y_test = np.array([label_to_index[label] for label in y_test])

    results = {}

    for model_type in model_types:
        if model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "knn":
            model = KNeighborsClassifier(n_neighbors=100)
        elif model_type == "xgboost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        else:
            raise ValueError(f"Modelo {model_type} no soportado.")

        model.fit(embeddings_train, y_train)
        y_pred = model.predict(embeddings_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_test, y_pred, labels=range(len(unique_labels)))

        results[model_type] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "conf_matrix": conf_matrix
        }

        print(
            f"{model_type} -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # Visualización de la Matriz de Confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt=".0f", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Matriz de Confusión - {model_type}")
        plt.show()

    return results


def predict_without_labels(train_embeddings, train_labels, test_embeddings, model_type="logistic_regression"):
    print("Entrenando modelo y realizando predicciones sin ground truth...")
    train_labels = np.concatenate([np.array(lbl).flatten() for lbl in train_labels])
    train_embeddings = np.vstack(train_embeddings)

    unique_labels = np.unique(train_labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.array([label_to_index[label] for label in train_labels])

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"Modelo {model_type} no soportado.")
    model.fit(train_embeddings, y_train)
    predictions = model.predict(test_embeddings)
    probabilities = model.predict_proba(test_embeddings)

    for key in label_to_index.keys():
        sum_predictions = np.sum(predictions==label_to_index[key] )
        print(f"for class {key}: {sum_predictions}")
    # Visualización de las predicciones
    plt.figure(figsize=(8, 6))
    plt.hist(np.max(probabilities, axis=1), bins=20, alpha=0.7, color='blue')
    plt.xlabel("Máxima Probabilidad de Predicción")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de Confianza en Predicciones")
    plt.show()

    return predictions, probabilities