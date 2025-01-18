import io
import json
from os.path import join as opj
from collections import OrderedDict

import sqlitedict
import torch
from torchvision import models
import torchvision.transforms as transforms
from torchvision.models import RegNet_X_800MF_Weights
from PIL import Image
from flask import Flask, jsonify, request
from tqdm import tqdm
import numpy as np

from train import get_top5_similarity
from model_utils import make_model

app = Flask(__name__)
app.json.sort_keys = False

transforms = RegNet_X_800MF_Weights.DEFAULT.transforms() 
model = make_model("./weights/backbone.pth")

print("Reading db...")
with sqlitedict.SqliteDict('./db_dump.sqlite') as db:
    filenames = list(db.keys())
    embeddings = np.stack([db[key] for key in filenames])
print(f"Readed {len(filenames)} records. Embeddings shape: {embeddings.shape}")

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transforms(image)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes).unsqueeze(0)
    emb = model.forward(tensor).detach().numpy()
    top5 = get_top5_similarity(emb, embeddings)
    ret_dict = OrderedDict()
    for k, v in top5.items():
        ret_dict[filenames[k]] = round(float(v),2)
    print(ret_dict)
    return ret_dict


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        return jsonify(get_prediction(image_bytes=img_bytes))

if __name__ == '__main__':
    app.run(host="0.0.0.0")