import os
import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from PIL import Image, ImageFont, ImageDraw, ImageEnhance

from model import SSD300
from data import TextDataset, MyTransform

threshold = 0
min_score = 0.2
max_overlap = 0.1
top_k = 200

box_color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 200, 0)]
text_color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 200, 0)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_parser():
    parser = argparse.ArgumentParser(description="Test Single Shot MultiBox Detector")
    parser.add_argument('--root', type=str, default='./manga109')
    parser.add_argument('--input', type=str, default='./manga109')
    parser.add_argument('--model', type=str, default='./models')
    parser.add_argument('--save', type=str, default='./result')
    return parser

def test(model, transform, original_image):
    model.eval()
    
    annotated_image = None
    
    image = transform(original_image)
    image = image.to(device)
    
    with torch.no_grad():
        predicted_locs, predicted_scores = model(image.unsqueeze(0))

        # Detect objects in SSD output ---- det_labels, 
        det_boxes, det_scores = model.detect_objects(
            predicted_locs, 
            predicted_scores, 
            min_score=min_score,
            max_overlap=max_overlap, 
            top_k=top_k)

        det_boxes  = det_boxes[0].cpu()
        # det_labels = det_labels[0].cpu()
        det_scores = det_scores[0].cpu()
        
        # ---- det_labels.shape, 
        print(det_boxes.shape, det_scores.shape)

        # Transform to original image dimensions
        original_dims = torch.FloatTensor(
            [
                original_image.width, 
                original_image.height, 
                original_image.width, 
                original_image.height
            ]).unsqueeze(0)

        det_boxes = det_boxes * original_dims
        
        # Annotate
        annotated_image = original_image
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.truetype("./consola.ttf", 15)

        for c in range(det_boxes.size(0)):
            class_det_boxes = det_boxes[c]
            # class_det_labels = det_labels[c]
            class_det_scores = det_scores[c]
            box_size = class_det_boxes.size(0)
            if box_size > 0:
                # to original size of box
                class_det_boxes = class_det_boxes * original_dims
                for i in range(box_size):
                    if class_det_scores[i] >= threshold:      
                        box_location = class_det_boxes[i].tolist()
                        draw.rectangle(xy=box_location, outline=box_color[c])
                        draw.text([box_location[0], box_location[1] - 15.], f'{class_det_scores[i].item()}', text_color[c], font=font)
        del draw
        
    model.train()
    return annotated_image
    
if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    obj = torch.load(args.model)
    model = obj['model']
    model.to(device)
    
    transform = MyTransform()
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    # #
    # root_folder = args.root
    # folder_list = os.listdir(root_folder)
    # l = len(folder_list)
    # for i in range(0, l):
    #     file_folder = folder_list[i]
    #     file_root = os.path.join(root_folder, file_folder)
    #     save_dir = os.path.join(args.save, f'{file_folder}')
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     for file in os.listdir(file_root):
    #         file_dir = os.path.join(file_root, file)
    #         original_image = Image.open(file_dir, mode='r').convert('RGB')
    #         result = test(model, transform, original_image)
    #         result.save(os.path.join(save_dir, f'{file}'), 'JPEG')
    
    # 測試單一資料
    original_image = Image.open(args.input, mode='r').convert('RGB')
    result = test(model, transform, original_image)
    result.save(os.path.join(args.save, 'test.jpg'), 'JPEG')
    
    