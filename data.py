import torch
from torch.utils.data import Dataset

import os
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

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
    def __init__(self, root_folder, model_type='ssd-fork', transforms=None):
        self.transforms = transforms

        self.default_labels = ['frame', 'text', 'face', 'body']
        self.imgs = []
        self.boxes = []
        self.labels = []

        self.model_type = model_type
        self.preprocessing(root_folder)
        
    def preprocessing(self, root_folder):
        # take manga109 input format
        xml_folder = os.path.join(root_folder, 'annotations')
        img_folder = os.path.join(root_folder, 'images')
        
        # construct dataset
        for annoXml in os.listdir(xml_folder):
            tree = ET.parse(os.path.join(xml_folder, annoXml))
            root = tree.getroot()

            manga_name = root.attrib['title']
            manga_folder = os.path.join(img_folder, manga_name)
            
            # parse 'pages' element
            for child in root:
                if child.tag == 'pages':
                    for page in child:
                        img_index = page.attrib['index']
                        img_path = os.path.join(manga_folder, img_index.zfill(3) + '.jpg')

                        # if model_type == 'ssd-fork':
                        #     # create box set
                        #     boxes = {}
                        #     for lbls in self.default_labels:
                        #         boxes[lbls] = []
                        # else:
                        boxes = []
                        labels = []

                        for element in page:
                            attr = element.attrib
                            if int(attr['xmin']) < int(attr['xmax']) and int(attr['ymin']) < int(attr['ymax']):
                                try:
                                    labelIndex = self.default_labels.index(element.tag)
                                    boxes.append([int(attr['xmin']), int(attr['ymin']), int(attr['xmax']), int(attr['ymax'])])
                                    labels.append(labelIndex)
                                except:
                                    print(f'without label type {element.tag}')
                                    continue

                        if len(boxes) > 0:
                            self.imgs.append(img_path)
                            self.boxes.append(boxes)
                            self.labels.append(labels)

            # break

    """
    # return [cx, cy, w, h]
    # cx    中心x
    # cy    中心y
    # w     width
    # h     height
    def bbox_to_ssd_format(self, index):
        # original box
        oboxes = self.boxes[index]
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
    """

    def __getitem__(self, index):
        # load image
        img_path = self.imgs[index]
        img = Image.open(img_path, mode='r').convert('RGB')
        
        boxes = torch.FloatTensor(self.boxes[index])
        # 0 mean negative, so we add one to all label index
        labels = torch.LongTensor(self.labels[index]) + 1

        img, boxes = self.transforms(img, boxes)

        if self.model_type == 'ssd-fork':
            boxes_fork = []
            labels_fork = []
            for c in range(len(self.default_labels)):
                pick = labels == (c + 1)
                boxes_fork.append(boxes[pick])
                # careful to the label index
                labels_fork.append(torch.ones(labels[pick].shape))
            boxes = boxes_fork
            labels = labels_fork

        return (img, boxes, labels)

    def __len__(self):
        return len(self.imgs)

def my_collate_fn(batch):
    imgs = list()
    boxes = list()
    labels = list()

    for b in batch:
        imgs.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])

    imgs = torch.stack(imgs, dim=0)

    return imgs, boxes, labels
