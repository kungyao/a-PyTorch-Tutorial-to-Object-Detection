import torch
from torch.utils.data import Dataset

import os
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

dataRange = {
    'train' : [0, 90],
    'test' : [90, 109],
}

# normalize image width and height from 0 to 1 by input dims
class MyTransform(object):
    def __call__(self, image, bboxes = None, dims=(300,300)):
        new_image = FT.resize(image, dims)
        new_image = FT.to_tensor(new_image)
        
        if bboxes == None:
            return new_image
        
        old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
        new_boxes = bboxes / old_dims
        
        return new_image, new_boxes

class TextDataset(Dataset):
    def __init__(self, root_folder, phase = 'train', transforms = None):
        self.root_folder = root_folder
        self.transforms = transforms

        # self.imgs = list(sorted(os.listdir(os.path.join(self.root_folder, 'imgs'))))
        # self.jsons = list(sorted(os.listdir(os.path.join(self.root_folder, 'jsons'))))
        
        self.phase = phase
        # self.dataRange = dataRange[phase]
        self.dataRange = dataRange['train']
        
        self.imgs = []
        self.text_boxes = []

        self.preprocessing()
        
    def preprocessing(self):
        xml_folder = os.path.join(self.root_folder, 'annotations')
        img_folder = os.path.join(self.root_folder, 'images')
        
        # construct training dataset
        listdir = os.listdir(xml_folder)
        for i in range(self.dataRange[0], self.dataRange[1]):
            xml = listdir[i]
            xml_path = os.path.join(xml_folder, xml)
            
            tree = ET.parse(xml_path)
            root = tree.getroot()

            manga_name = root.attrib['title']
            manga_folder = os.path.join(img_folder, manga_name)
            
            # parse 'pages' element
            for child in root:
                if child.tag == 'pages':
                    for page in child:
                        img_index = page.attrib['index']
                        img_path = os.path.join(manga_folder, img_index.zfill(3) + '.jpg')
                        box = []
                        for element in page:
                            if element.tag == 'text':
                                attr = element.attrib
                                if int(attr['xmin']) < int(attr['xmax']) and int(attr['ymin']) < int(attr['ymax']):
                                    box.append([int(attr['xmin']), int(attr['ymin']), int(attr['xmax']), int(attr['ymax'])])
                        if len(box) > 0:
                            self.imgs.append(img_path)
                            self.text_boxes.append(box)
    # return        
    # [cx, cy, w, h]
    # cx    中心x
    # cy    中心y
    # w     width
    # h     height
    def bbox_to_ssd_format(self, index):
        # original box
        oboxes = self.text_boxes[index]
        bboxes = []
        for obox in oboxes:
            bbox = [
                (obox[0] + obox[2]) / 2,
                (obox[1] + obox[3]) / 2,
                obox[2] - obox[0],
                obox[3] - obox[1]
            ]
            bboxes.append(bbox)
        return bboxes

    def __getitem__(self, index):
        # load image
        img_path = self.imgs[index]
        img = Image.open(img_path, mode='r').convert('RGB')
        # img = Image.open(img_path, mode='r').convert('1', dither=None)
        
        # test mode need original image and scaled image
        # origin is used to draw the result box
        # scaled one is used to predict by model
        if self.phase == 'test':
            transImage = self.transforms(img)
            return (to_tensor(img), transImage)
        
        boxes = torch.FloatTensor(self.text_boxes[index].copy())
        labels = torch.ones((len(boxes)), dtype=torch.int64)

        img, boxes = self.transforms(img, boxes)

        return (img, boxes, labels)

    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch):
        if self.phase == 'train':
            images = list()
            boxes = list()
            labels = list()
            
            for b in batch:
                images.append(b[0])
                boxes.append(b[1])
                labels.append(b[2])        
            
            images = torch.stack(images, dim=0)
            
            return images, boxes, labels
            
        elif self.phase == 'test':
            oriImage = list()
            transImage = list()
            
            for b in batch:
                oriImage.append(b[0])
                transImage.append(b[1])    
            
            images = torch.stack(images, dim=0)
            transImage = torch.stack(transImage, dim=0)
            
            return oriImage, transImage
