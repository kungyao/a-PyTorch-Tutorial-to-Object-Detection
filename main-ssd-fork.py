import os
import time
from argparse import ArgumentParser

from model_fork import SSD300Fork
from model import SSD300

import torch
from PIL import Image
import torchvision.transforms.functional as TF

path = "v2-e82cb9a688c9cba9265e1044c5159d7b_hd.jpg"
img = Image.open(path, mode='r').convert('RGB')
img = img.resize((300, 300))
img = TF.to_tensor(img)

imgs = torch.stack([img], dim=0)

model = SSD300Fork(2)
# model = SSD300()
print(model)

predicted_locs, predicted_scores = model(imgs)
print(predicted_locs.shape, predicted_scores.shape)

# threshold = 0.5
min_score = 0.5
max_overlap = 0.5
top_k = 200

det_boxes, det_labels, det_scores = model.detect_objects(
    predicted_locs, 
    predicted_scores, 
    min_score=min_score,
    max_overlap=max_overlap, 
    top_k=top_k)

