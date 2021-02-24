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

# threshold = 0.5
min_score = 0.5
max_overlap = 0.5
top_k = 200

box_color = (255, 0, 0)
text_color = (0, 255, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_parser():
    parser = argparse.ArgumentParser(description="Test Single Shot MultiBox Detector")
    parser.add_argument('--root', type=str, default='./manga109')
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

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = model.detect_objects(
            predicted_locs, 
            predicted_scores, 
            min_score=min_score,
            max_overlap=max_overlap, 
            top_k=top_k)

        det_boxes  = det_boxes[0].cpu()
        
        det_labels = det_labels[0].cpu()
        det_scores = det_scores[0].cpu()
        
        if det_labels.size(0) == 1:
            # only train for a class
            if det_labels[0].item() == 0:
                model.train()
                return original_image
        
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
        
        for i in range(det_boxes.size(0)):
            # Boxes
            box_location = det_boxes[i].tolist()
            draw.rectangle(xy=box_location, outline=box_color)
            
            # Text
            draw.text([box_location[0], box_location[1] - 15.], f'{det_scores[i].item()}', text_color, font=font)
            
        del draw
        
    model.train()
    return annotated_image
    
if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    
    num_classes = 2
    
    # text and another
    model = SSD300(num_classes)
    epoch = 14
    save_path = os.path.join(args.model, f'epoch_{epoch}.pt')
    obj = torch.load(save_path)
    # model = obj['model']
    model.load_state_dict(obj['model'])
    model.to(device)
    
    transform = MyTransform()
    root_folder = args.root
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    #
    folder_list = os.listdir(root_folder)
    l = len(folder_list)
    for i in range(0, l):
        file_folder = folder_list[i]
        file_root = os.path.join(root_folder, file_folder)
        save_dir = os.path.join(args.save, f'{file_folder}')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for file in os.listdir(file_root):
            file_dir = os.path.join(file_root, file)
            original_image = Image.open(file_dir, mode='r').convert('RGB')
            result = test(model, transform, original_image)
            result.save(os.path.join(save_dir, f'{file}'), 'JPEG')
    
    # # 測試單一資料
    # file_dir = 'D:/Manga/Manga109/images/MeteoSanStrikeDesu/015.jpg'
    # original_image = Image.open(file_dir, mode='r').convert('RGB')
    # result = test(model, transform, original_image)
    # result.save(os.path.join('C:/Users/masa1/Desktop/test.jpg'), 'JPEG')
    
    