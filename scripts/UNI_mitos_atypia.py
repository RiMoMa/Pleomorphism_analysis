
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
import torch
import os
HF_TOKEN = os.getenv("HF_TOKEN")

login(token=HF_TOKEN)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

# pretrained=True needed to load UNI2-h weights (and download weights for the first time)
timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()

import torch
import numpy as np
import os
from torchvision import transforms
from PIL import Image

# Configure UNI model
# from uni import get_encoder
# model, transform = get_encoder(enc_name='uni2-h', device="cuda")
# from PIL import Image


# Define dataset path
root_dir = 'data/ATYPIA_classes/' # Adjust the path accordingly
embeddings_dir = "data_processed/embeddings_UNI_atypia/"
os.makedirs(embeddings_dir, exist_ok=True)
# Process images in each class folder
for class_name in sorted(os.listdir(root_dir)):
    class_path = os.path.join(root_dir, class_name)
    embedding_path = os.path.join(embeddings_dir, class_name)
    os.makedirs(embedding_path, exist_ok=True)
    if os.path.isdir(class_path):
        for img_name in sorted(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)
            path_img_embedding= os.path.join(embedding_path, f"{img_name}.npy")
            if os.path.exists(path_img_embedding):
                continue
            try:
                image = Image.open(img_path).convert("RGB")
                image = transform(image).unsqueeze(
                    dim=0)  # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization (ImageNet parameters)
                with torch.inference_mode():
                    feature_emb = model(image)  # Extracted features (torch.Tensor) with shape [1, 1536]
                # Extract feature embeddings
                np.save(path_img_embedding, feature_emb.detach().numpy().flatten())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print(f"Embeddings saved in {embeddings_dir}")


import umap
import matplotlib
matplotlib.use('TkAgg')  # Usa TkAgg en lugar del backend de PyCharm
import matplotlib.pyplot as plt
  # O prueba 'Agg', 'Qt4Agg', 'GTK3Agg'
import matplotlib.pyplot as plt


# Load saved embeddings
embeddings_dir = "data_processed/embeddings_UNI_atypia/"
embeddings = []
labels = []

for class_name in sorted(os.listdir(embeddings_dir)):
    class_path = os.path.join(embeddings_dir, class_name)
    if os.path.isdir(class_path):
        for file in sorted(os.listdir(class_path)):
            if file.endswith(".npy"):
                embeddings.append(np.load(os.path.join(class_path, file)))
                labels.append(class_name)

# Convert to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# Assign colors based on class labels
unique_labels = list(set(labels))
color_map = {label: idx for idx, label in enumerate(unique_labels)}
colors = [color_map[label] for label in labels]

# Plot the 2D embeddings
plt.figure(figsize=(8, 6))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.7,s=5, cmap='viridis')
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Projection of UNI Embeddings by Class")
cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
cbar.set_ticklabels(unique_labels)
cbar.set_label("Class Index")
plt.show()

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Definir número de folds
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Convertir etiquetas a valores numéricos
unique_labels = np.unique(labels)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y = np.array([label_to_index[label] for label in labels])

# Almacenar métricas
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
conf_matrix_accumulated = np.zeros((len(unique_labels), len(unique_labels)))

# Realizar 5-Fold Cross Validation
for train_idx, test_idx in skf.split(embeddings, y):
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Entrenar modelo de Regresión Logística
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predecir en conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular métricas
    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred, average='macro'))
    recall_list.append(recall_score(y_test, y_pred, average='macro'))
    f1_list.append(f1_score(y_test, y_pred, average='macro'))
    conf_matrix_accumulated += confusion_matrix(y_test, y_pred, labels=range(len(unique_labels)))

# Mostrar resultados promedio
print("Resultados de 5-Fold Cross Validation:")
print(f"Accuracy: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
print(f"Precision: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
print(f"Recall: {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
print(f"F1-Score: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")

# Mostrar matriz de confusión acumulada
print("Matriz de Confusión Acumulada:")
print(conf_matrix_accumulated)
