from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, Response, HTMLResponse
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

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import torch

import zipfile

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
cityscapes_classes_8 = list(cats.keys())

cityscapes_palette_8 = [
    (0, 0, 0),           # 0 - void        → noir
    (128, 64, 128),      # 1 - flat        → violet (road)
    (70, 70, 70),        # 2 - construction→ gris
    (255, 165, 0),       # 3 - object      → orange
    (107, 142, 35),      # 4 - nature      → vert
    (70, 130, 180),      # 5 - sky         → bleu ciel
    (220, 20, 60),       # 6 - human       → rouge
    (0, 0, 142)          # 7 - vehicle     → bleu foncé
]
cmap_cityscapes_8 = ListedColormap(np.array(cityscapes_palette_8) / 255.0)

def apply_palette(mask, palette):
    color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        color_seg[mask == class_id] = color
    return color_seg

def blend(image, mask, palette):
    color_seg = apply_palette(mask, palette)
    # Fusion image originale (RGB) et masque coloré
    blended = 0.5 * np.array(image, dtype=float) + 0.5 * color_seg
    return blended.astype(np.uint8)

cityscapes_19_to_8_mapping = {
    0: 1, 1: 1,               # flat
    2: 2, 3: 2, 4: 2,         # construction
    5: 3, 6: 3, 7: 3,         # object
    8: 4, 9: 4,               # nature
    10: 5,                    # sky
    11: 6, 12: 6,             # human
    13: 7, 14: 7, 15: 7, 16: 7, 17: 7, 18: 7  # vehicle
}
def remap_cityscapes_to_8classes(mask):
    remapped = np.full_like(mask, fill_value=255)
    for train_id, new_id in cityscapes_19_to_8_mapping.items():
        remapped[mask == train_id] = new_id
    return remapped

def render_to_buffer(image, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return buf

def generate_mask_overlay(img, mask):
    return render_to_buffer(blend(img, mask, cityscapes_palette_8))

# ------------------------------------------------------------------------------

app = FastAPI()

DATA_DIR = "data/images"
MASK_DIR = "data/masks"
img_height, img_width, n_classes = 256, 256, len(cityscapes_classes_8)

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

    # Récupère uniquement les identifiants d’images (sans extension)
    ids = [f.split(".")[0] for f in files if f.endswith(".png")]

    # Retourne la liste des IDs sous forme JSON
    return JSONResponse(content={"ids": ids})

# Endpoint pour récupérer une image par son ID
@app.get("/list_images/{image_id}")
async def fetch_image_by_id(image_id: str):
    # Construit le chemin vers l'image PNG correspondante
    image_path = os.path.join(DATA_DIR, f"{image_id}.png")
    img = np.array(Image.open(image_path).convert("RGB"), dtype=int)

    # Affichage du masque avec une colormap
    buf = render_to_buffer(img)

    # Vérifie que l'image existe
    if not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"error": "Image not found"})

    # Retour de l'image en réponse HTTP
    return Response(content=buf.getvalue(), media_type="image/png")

# Fonction pour charger, redimensionner et encoder une image de masque en format one-hot
def load_and_encode_mask(mask_path):
    # Chargement du masque en niveaux de gris (format PIL)
    mask = image.load_img(mask_path, color_mode='grayscale')

    # Conversion en tableau numpy (H, W, 1)
    mask = image.img_to_array(mask)

    # Conversion en entier (uint8)
    mask = tf.cast(mask, tf.uint8).numpy()

    # Fonction interne pour encoder le masque en one-hot
    def encode_mask(mask_img):
        mask_img = np.squeeze(mask_img).astype(np.uint8)  # Suppression du canal inutile
        height, width = mask_img.shape
        new_mask = np.zeros((height, width, n_classes), dtype=np.float32)
        for label, class_idx in cat_mapping.items():
            new_mask[mask_img == label, class_idx] = 1.0
        return new_mask

    encoded_mask = encode_mask(mask)

    # Retourne la version "argmax" (classe dominante par pixel), sans resize
    return np.argmax(encoded_mask, axis=-1)

# Endpoint FastAPI pour récupérer une image du masque colorisée
@app.get("/get_mask/{image_id}")
async def get_mask(image_id: str):
    mask_path = os.path.join(MASK_DIR, f"{image_id}.png")
    image_path = os.path.join(DATA_DIR, f"{image_id}.png")
    img = np.array(Image.open(image_path).convert("RGB"), dtype=int)

    # Vérifie si le fichier de masque existe
    if not os.path.exists(mask_path):
        return JSONResponse(status_code=404, content={"error": "Mask not found"})

    # Chargement et traitement du masque
    mask = load_and_encode_mask(mask_path)

    # Affichage du masque avec une colormap
    buf = generate_mask_overlay(img, mask)

    # Retour de l'image en réponse HTTP
    return Response(content=buf.getvalue(), media_type="image/png")

# Renvoie la liste des classes et leurs couleurs associées.
@app.get("/legend_data/")
def get_legend_data():
    legend = [
        {"name": name, "color": list(color)}
        for name, color in zip(cityscapes_classes_8, cityscapes_palette_8)
    ]
    return JSONResponse(content={"legend": legend})

# Renvoie les noms des modèles disponibles
@app.get("/available_models/")
def list_models():
    return {"models": list(models.keys())}

# ------------------------------------------------------------------------------

DILATEDNET_URL = "https://imgsegmodelstorage.blob.core.windows.net/models/dilatednet.keras"
DILATEDNET_PATH = "models/dilatednet.keras"

MASK2FORMER_URL = "https://imgsegmodelstorage.blob.core.windows.net/models/mask2former.zip"
MASK2FORMER_ZIP_PATH = "models/mask2former.zip"
MASK2FORMER_DIR = "models/mask2former/mask2former"

def download_models_if_needed():
    # --- DilatedNet ---
    os.makedirs(os.path.dirname(DILATEDNET_PATH), exist_ok=True)
    if not os.path.exists(DILATEDNET_PATH):
        print("Téléchargement du modèle DilatedNet...")
        with requests.get(DILATEDNET_URL, stream=True) as r:
            r.raise_for_status()
            with open(DILATEDNET_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Modèle DilatedNet téléchargé.")
    else:
        print("Modèle DilatedNet déjà présent.")

    # --- Mask2Former ---
    os.makedirs(MASK2FORMER_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(MASK2FORMER_DIR, "config.json")):
        print("Téléchargement du modèle Mask2Former...")
        with requests.get(MASK2FORMER_URL, stream=True) as r:
            r.raise_for_status()
            with open(MASK2FORMER_ZIP_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Extraction du modèle Mask2Former...")
        with zipfile.ZipFile(MASK2FORMER_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MASK2FORMER_DIR)
        os.remove(MASK2FORMER_ZIP_PATH)
        print("Modèle Mask2Former téléchargé et extrait.")
    else:
        print("Modèle Mask2Former déjà extrait.")

# Télécharger le modèle si nécessaire
download_models_if_needed()

# Charger le modèle
def load_model(model_name):
    if model_name == "dilatednet":

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

        model = tf.keras.models.load_model(DILATEDNET_PATH, compile=False)
        model.compile(optimizer=Adam(1e-4), loss=total_loss, metrics=[dice_coeff, 'accuracy'])
        return model
    elif model_name == "mask2former_pretrained":
        model_id = "facebook/mask2former-swin-small-cityscapes-semantic"
        return Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
    elif model_name == "mask2former_finetuned":
        return Mask2FormerForUniversalSegmentation.from_pretrained(MASK2FORMER_DIR)

def load_processor(model_name):
    if model_name == "mask2former_pretrained":
        model_id = "facebook/mask2former-swin-small-cityscapes-semantic"
        return Mask2FormerImageProcessor.from_pretrained(model_id)
    elif model_name == "mask2former_finetuned":
        return Mask2FormerImageProcessor.from_pretrained(MASK2FORMER_DIR)


model_names = ['dilatednet', 'mask2former_finetuned', 'mask2former_pretrained']
models = {}
processors = {}

for model_name in model_names:
    models[model_name] = load_model(model_name)
    if model_name.startswith("mask2former"):
        processors[model_name] = load_processor(model_name)

# ------------------------------------------------------------------------------

def predict_mask(image_input, original_size, model_name: str = "dilatednet", from_file: bool = True):
    model = models.get(model_name)
    if model is None:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name == "dilatednet":
        # Prétraitement pour dilatednet
        img = image_input.resize((img_width, img_height))
        img_array = image.img_to_array(img) / 255.0
        img_array =  np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        reshaped = prediction[0].reshape((img_height, img_width, n_classes))
        resized_logits = tf.image.resize(reshaped, original_size, method="bilinear").numpy()
        # Calcul du masque final par argmax sur les logits (classe avec proba max)
        predicted_mask = np.argmax(resized_logits, axis=-1)
        return predicted_mask

    elif model_name.startswith("mask2former"):
        # Prétraitement pour mask2former
        processor = processors.get(model_name)
        if processor is None:
            raise ValueError("Processor for mask2former not loaded.")

        # Resize pour garder une référence à la taille originale
        inputs = processor(images=image_input, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-processing pour obtenir les prédictions en trainIds
        pred_mask = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[original_size]
        )[0].cpu().numpy()

        # Remapping → 8 classes
        if model_name=='mask2former_pretrained':
            pred_mask = remap_cityscapes_to_8classes(pred_mask)

        return np.array(pred_mask)

@app.post("/predict_from_file/")
async def predict_from_file(file: UploadFile = File(...),  model_name: str = Form(...)):
    print(f"Image reçue : {file.filename} — type : {file.content_type} — modèle : {model_name}")

    contents = file.file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Fichier image invalide."})

    original_size = img.size[::-1]

    predicted_mask = predict_mask(img, original_size, model_name=model_name, from_file=True)

    # Affichage du masque avec une colormap
    buf = generate_mask_overlay(img, predicted_mask)

    # Retour de l'image en réponse HTTP
    return Response(content=buf.getvalue(), media_type="image/png")

class PredictRequest(BaseModel):
    image_id: str
    model_name: str

@app.post("/predict_from_id/")
async def predict_from_id(req: PredictRequest):
    image_path = os.path.join(DATA_DIR, f"{req.image_id}.png")
    img = Image.open(image_path).convert("RGB")
    original_size = img.size[::-1]

    if not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"error": "Image not found"})

    predicted_mask = predict_mask(img, original_size, model_name=req.model_name, from_file=False)

    # Affichage du masque avec une colormap
    buf = generate_mask_overlay(img, predicted_mask)

    # Retour de l'image en réponse HTTP
    return Response(content=buf.getvalue(), media_type="image/png")
