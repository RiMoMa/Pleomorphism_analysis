import numpy as np
import os

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Desactiva la restricción

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
import torch
def extract_and_save_embeddings(data_path, embeddings_dir, model_name="hf-hub:MahmoodLab/UNI2-h"):
    HF_TOKEN = os.getenv("HF_TOKEN")
    print("Cargando modelo de embeddings...")
    login(
        token=HF_TOKEN)  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    # pretrained=True needed to load UNI2-h weights (and download weights for the first time)
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
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


    os.makedirs(embeddings_dir, exist_ok=True)
    embeddings_list = []
    labels_list = []
    names = []

    for class_name in sorted(os.listdir(data_path)):
        class_path = os.path.join(data_path, class_name)
        embedding_path = os.path.join(embeddings_dir, class_name)
        os.makedirs(embedding_path, exist_ok=True)

        if os.path.isdir(class_path):
            for img_name in sorted(os.listdir(class_path)):
                img_path = os.path.join(class_path, img_name)
                img_name_no_ext = os.path.splitext(img_name)[0]
                path_img_embedding = os.path.join(embedding_path, f"{img_name_no_ext}.npy")
                names.append(class_name+'_'+img_name_no_ext)
                if os.path.exists(path_img_embedding):
                    embeddings_list.append(np.load(path_img_embedding))
                    labels_list.append(class_name)

                    continue
                try:
                    image = Image.open(img_path).convert("RGB")

                    image = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
                    with torch.inference_mode():
                        feature_emb = model(image)  # [1, 1536]
                    embedding_array = feature_emb.detach().numpy().flatten()

                    np.save(path_img_embedding, embedding_array)
                    embeddings_list.append(embedding_array)
                    labels_list.append(class_name)
                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")
    print(f"Embeddings guardados en {embeddings_dir}")
    return np.array(embeddings_list), np.array(labels_list), names




