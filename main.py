from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import numpy as np
import requests
import os
import io

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
matplotlib.use('Agg')

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras import backend as K

from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
import torch


# ------------------------------------------------------------------------------
# Mapping des 34 classes → 8 grandes catégories
cats = {
    'void': [0, 1, 2, 3, 4, 5, 6],
    'flat': [7, 8, 9, 10],
    'construction': [11, 12, 13, 14, 15, 16],
    'object': [17, 18, 19, 20],
    'nature': [21, 22],
    'sky': [23],
    'human': [24, 25],
    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1],
}
cat_mapping = {i: idx for idx, (_, ids) in enumerate(cats.items()) for i in ids}

EIGHT_CLASS_COLORS = [
    (0, 0, 0),           # 0 - void
    (128, 64, 128),      # 1 - flat
    (70, 70, 70),        # 2 - construction
    (255, 165, 0),       # 3 - object
    (107, 142, 35),      # 4 - nature
    (70, 130, 180),      # 5 - sky
    (220, 20, 60),       # 6 - human
    (0, 0, 142)          # 7 - vehicle
]
class_names = ['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']
cmap = ListedColormap(np.array(EIGHT_CLASS_COLORS) / 255.0)


CITYSCAPES_TO_8_CLASSES = {
    0: 1, 1: 1,
    2: 2, 3: 2, 4: 2,
    5: 3, 6: 3, 7: 3,
    8: 4, 9: 4,
    10: 5,
    11: 6, 12: 6,
    13: 7, 14: 7, 15: 7, 16: 7, 17: 7, 18: 7
}
def remap_cityscapes_to_8classes(mask: np.ndarray) -> np.ndarray:
    remapped = np.full_like(mask, fill_value=255)
    for train_id, new_id in CITYSCAPES_TO_8_CLASSES.items():
        remapped[mask == train_id] = new_id
    return remapped


# ------------------------------------------------------------------------------

app = FastAPI()

DATA_DIR = "data/images"
MASK_DIR = "data/masks"
img_height, img_width, n_classes = 256, 256, len(class_names)

# ------------------------------------------------------------------------------

@app.get("/")
def home():
    return {"message": "Image Segmentation API"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")

# Endpoint pour lister toutes les images disponibles dans DATA_DIR
@app.get("/list_images/")
async def list_available_images():
    print("📂 Lecture des images dans :", DATA_DIR)

    # Liste tous les fichiers dans le dossier de données
    files = os.listdir(DATA_DIR)
    print("Fichiers trouvés :", files)

    # Récupère uniquement les identifiants d’images (sans extension)
    ids = [f.split(".")[0] for f in files if f.endswith(".png")]

    # Retourne la liste des IDs sous forme JSON
    return JSONResponse(content={"ids": ids})


# Endpoint pour récupérer une image par son ID
@app.get("/list_images/{image_id}")
async def fetch_image_by_id(image_id: str):
    # Construit le chemin vers l'image PNG correspondante
    image_path = os.path.join(DATA_DIR, f"{image_id}.png")

    # Vérifie que l'image existe
    if not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"error": "Image not found"})

    # Retourne l'image en tant que fichier PNG
    return FileResponse(image_path, media_type="image/png")


# Fonction pour charger, redimensionner et encoder une image de masque en format one-hot
def load_and_encode_mask(mask_path):
    # Chargement du masque en niveaux de gris
    mask = image.img_to_array(image.load_img(mask_path, color_mode='grayscale'))

    # Redimensionnement à la taille cible (img_height, img_width)
    mask = tf.image.resize(mask, (img_height, img_width), method='nearest')

    # Conversion en entier (uint8) pour traitement par numpy
    mask = tf.cast(mask, tf.uint8).numpy()

    # Fonction interne pour encoder le masque en one-hot
    def encode_mask(mask_img):
        mask_img = np.squeeze(mask_img).astype(np.uint8)  # Suppression de la dimension canal
        new_mask = np.zeros((img_height, img_width, n_classes), dtype=np.float32)
        for label, class_idx in cat_mapping.items():  # Mapping label -> indice classe
            new_mask[mask_img == label, class_idx] = 1.0
        return new_mask

    encoded_mask = encode_mask(mask)

    # Retourne la version "argmax" (classe dominante par pixel)
    return np.argmax(encoded_mask, axis=-1)

# Endpoint FastAPI pour récupérer une image du masque colorisée
@app.get("/get_mask/{image_id}")
async def get_mask(image_id: str):
    mask_path = os.path.join(MASK_DIR, f"{image_id}.png")

    # Vérifie si le fichier de masque existe
    if not os.path.exists(mask_path):
        return JSONResponse(status_code=404, content={"error": "Mask not found"})

    # Chargement et traitement du masque
    mask = load_and_encode_mask(mask_path)

    # Affichage du masque avec une colormap
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=n_classes - 1)
    ax.axis("off")

    # Enregistrement de l'image dans un buffer mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Retour de l'image en réponse HTTP
    return Response(content=buf.getvalue(), media_type="image/png")

# ------------------------------------------------------------------------------

# Dice coefficient and loss
def dice_coeff(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coeff(y_true, y_pred)

def total_loss(y_true, y_pred):
    ce = CategoricalCrossentropy()(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return ce + 3 * dl

# ------------------------------------------------------------------------------

DILATEDNET_URL = "https://imgsegmodelstorage.blob.core.windows.net/models/dilatednet.keras"
DILATEDNET_PATH = "models/dilatednet.keras"

def download_delatednet_model_if_needed():
    os.makedirs(os.path.dirname(DILATEDNET_PATH), exist_ok=True)

    if not os.path.exists(DILATEDNET_PATH):
        print("Téléchargement du modèle depuis Azure Blob Storage...")
        with requests.get(DILATEDNET_URL, stream=True) as r:
            r.raise_for_status()
            with open(DILATEDNET_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Modèle téléchargé avec succès !")
    else:
        print("Modèle déjà présent.")

# Télécharger le modèle si nécessaire
download_delatednet_model_if_needed()

# Charger le modèle

def load_model(model_name):
    if model_name == "dilatednet":
        model = tf.keras.models.load_model(DILATEDNET_PATH, compile=False)
        model.compile(optimizer=Adam(1e-4), loss=total_loss, metrics=[dice_coeff, 'accuracy'])
        return model
    elif model_name == "mask2former":
        model_id = "facebook/mask2former-swin-small-cityscapes-semantic"
        return Mask2FormerForUniversalSegmentation.from_pretrained(model_id)


def load_processor(model_name):
    if model_name == "mask2former":
        model_id = "facebook/mask2former-swin-small-cityscapes-semantic"
        processor = AutoImageProcessor.from_pretrained(model_id)
        return processor

model_names = ['dilatednet', 'mask2former']
models = {}
processors = {}

for model_name in model_names:
    models[model_name] = load_model(model_name)
    if model_name == "mask2former":
        processors[model_name] = load_processor(model_name)

# ------------------------------------------------------------------------------

def preprocess_image_from_file(file: UploadFile):
    contents = file.file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    pil_img = pil_img.resize((img_width, img_height))
    img_array = image.img_to_array(pil_img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_image_from_path(path: str):
    pil_img = Image.open(path).convert("RGB")
    pil_img = pil_img.resize((img_width, img_height))
    img_array = image.img_to_array(pil_img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_mask(image_input, model_name: str = "dilatednet", from_file: bool = True):
    model = models.get(model_name)
    if model is None:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name == "dilatednet":
        # Prétraitement pour dilatednet
        if from_file:
            img_array = preprocess_image_from_file(image_input)
        else:
            img_array = preprocess_image_from_path(image_input)

        prediction = model.predict(img_array)
        reshaped = prediction[0].reshape((img_height, img_width, n_classes))
        predicted_mask = np.argmax(reshaped, axis=-1)
        return predicted_mask

    elif model_name == "mask2former":
        # Prétraitement pour mask2former
        if from_file:
            pil_img = Image.open(io.BytesIO(image_input.file.read())).convert("RGB")
        else:
            pil_img = Image.open(image_input).convert("RGB")

        processor = processors.get("mask2former")
        if processor is None:
            raise ValueError("Processor for mask2former not loaded.")

        # Resize pour garder une référence à la taille originale
        target_size = pil_img.size[::-1]  # (H, W)
        inputs = processor(images=pil_img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-processing pour obtenir les prédictions en trainIds
        pred_mask = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[target_size]
        )[0].cpu().numpy()

        # Remapping → 8 classes
        pred_remapped = remap_cityscapes_to_8classes(pred_mask)

        # Resize pour correspondre à (img_height, img_width)
        pred_resized = Image.fromarray(pred_remapped.astype(np.uint8)).resize((img_width, img_height), resample=Image.NEAREST)
        return np.array(pred_resized)


# ------------------------------------------------------------------------------

@app.post("/predict_from_file/")
async def predict_from_file(file: UploadFile = File(...), model_name: str = "dilatednet"):
    print(f"Image reçue : {file.filename} — type : {file.content_type}")
    predicted_mask = predict_mask(file, model_name=model_name, from_file=True)
    return JSONResponse(content={"mask": predicted_mask.tolist()})

class PredictRequest(BaseModel):
    image_id: str
    model_name: Optional[str] = "dilatednet"

@app.post("/predict_from_id/")
async def predict_from_id(req: PredictRequest):
    image_path = os.path.join(DATA_DIR, f"{req.image_id}.png")
    if not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"error": "Image not found"})

    predicted_mask = predict_mask(image_path, model_name=req.model_name, from_file=False)
    return JSONResponse(content={"mask": predicted_mask.tolist()})
