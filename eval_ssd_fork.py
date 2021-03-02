import os
import argparse
from pprint import PrettyPrinter

import torch
from torch.utils.data import DataLoader

from utils import calculate_mAP_fork
from data import TextDataset, MyTransform, my_collate_fn
from test_ssd_fork import visual_detection_result

min_score = 0.18
max_overlap = 0.2
top_k = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(test_loader, model, if_visualize, output, transforms):
    model.eval()

    n_classes = model.n_classes
    itertaion = 0
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    with torch.no_grad():
        for nbatch, (images, bboxes, labels) in enumerate(test_loader):
            # if visualize, loader will return pil type of image
            if if_visualize:
                ori_imgs = []
                for i in range(len(images)):
                    img = images[i]
                    ori_imgs.append(img)
                    images[i] = transforms(img)
                images = torch.stack(images, dim=0)
                images = images.to(device)
            # return tensor type of image
            else:
                images = images.to(device)

            # from (n, class, {}) to (n, class * {})
            for nb in range(len(bboxes)):
                bboxes[nb] = torch.cat(bboxes[nb], dim=0).to(device)
                class_label = list()
                for c in range(n_classes):
                    class_label.extend([c + 1] * len(labels[nb][c]))
                labels[nb] = torch.LongTensor(class_label).to(device)

            predicted_locs, predicted_scores = model(images)

            # Detect objects from SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs, 
                predicted_scores, 
                min_score=min_score,
                max_overlap=max_overlap, 
                top_k=top_k)

            if if_visualize:
                itertaion = visual_detection_result(n_classes, ori_imgs, det_boxes_batch, det_labels_batch, det_scores_batch, output, itertaion)

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(bboxes)
            true_labels.extend(labels)

    APs, mAP = calculate_mAP_fork(det_boxes, det_labels, det_scores, true_boxes, true_labels)
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)

    model.train()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./manga109')
    parser.add_argument('--output', type=str, default='./result')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batchsize', type=int, default=5)
    parser.add_argument('--manga_name', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    # Good formatting when printing the APs for each class and mAP
    pp = PrettyPrinter()
    
    args = get_args()

    obj = torch.load(args.model)
    model = obj['model']

    if args.manga_name != None:
        args.output = os.path.join(args.output, args.manga_name)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.visualize:
        mytransforms = MyTransform()
        test_dataset = TextDataset(args.root, model_type='ssd-fork', transforms=None, specific_manga=args.manga_name)
    else:
        mytransforms = None
        test_dataset = TextDataset(args.root, model_type='ssd-fork', transforms=MyTransform(), specific_manga=args.manga_name)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, collate_fn=my_collate_fn, num_workers=4, pin_memory=True)

    evaluate(test_loader, model, args.visualize, args.output, mytransforms)
