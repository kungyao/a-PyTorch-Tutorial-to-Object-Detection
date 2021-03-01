from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
import torchvision

from model import VGGBase, AuxiliaryConvolutions, PredictionConvolutions

class SSD300Fork(nn.Module):
    def __init__(self, n_classes):
        super(SSD300Fork, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()

        # ssd300-fork : foreach class need a detection layer
        self.pred_convs = []
        for _ in range(n_classes):
            self.pred_convs.append(PredictionConvolutions(2))
        self.pred_convs = nn.ModuleList(self.pred_convs)

        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Rescale conv4_3 after L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        # (num_of_class, N, 8732, 4)
        ret_locs = []
        # (num_of_class, N, 8732, 2)
        ret_classes_scores = []
        for detection_layer in self.pred_convs:
            # (N, 8732, 4), (N, 8732, 2)
            locs, classes_scores = detection_layer(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)
            ret_locs.append(locs)
            ret_classes_scores.append(classes_scores)

        # (N, num_of_class, 8732, 4)
        ret_locs = torch.stack(ret_locs, dim=0).permute(1, 0, 2, 3)
        # (N, num_of_class, 8732, 2)
        ret_classes_scores = torch.stack(ret_classes_scores, dim=0).permute(1, 0, 2, 3)
        return ret_locs, ret_classes_scores

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)

        # do softmax to 3rd-dimension
        predicted_scores = F.softmax(predicted_scores, dim=3)  # (N, n_classes, 8732, 2)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        # total box size need to be same
        assert n_priors == predicted_locs.size(2) == predicted_scores.size(2)

        # Check for each class
        for i in range(batch_size):
            # Lists to store boxes and scores for this image
            image_boxes = list()
            # image_labels = list()
            image_scores = list()
            for c in range(self.n_classes): 
                # Decode object coordinates from the form we regressed predicted boxes to
                decoded_locs = cxcy_to_xy(
                    gcxgcy_to_cxcy(predicted_locs[i][c], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates
                max_scores, best_label = predicted_scores[i][c].max(dim=1)  # (8732)
                #-----------------------------------------------------------------------------------
                # Check for each class
                # for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][c][:, 1]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()

                if n_above_min_score != 0:
                    class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                    class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)
                    print(i, c, n_above_min_score)
                    # Sort predicted boxes and scores by scores
                    class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                    class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                    # Find the overlap between predicted boxes
                    overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                    # Non-Maximum Suppression (NMS)

                    # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                    # 1 implies suppress, 0 implies don't suppress
                    suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                    # Consider each box in order of decreasing scores
                    for box in range(class_decoded_locs.size(0)):
                        # If this box is already marked for suppression
                        if suppress[box] == 1:
                            continue

                        # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                        # Find such boxes and update suppress indices
                        # print(overlap[box], (overlap[box] > max_overlap).type(torch.uint8))
                        # suppress = torch.max(suppress, (overlap[box] > max_overlap).type(torch.uint8))
                        suppress = torch.max(suppress, (overlap[box] > max_overlap).type(torch.uint8))
                        # The max operation retains previously suppressed boxes, like an 'OR' operation

                        # Don't suppress this box, even though it has an overlap of 1 with itself
                        suppress[box] = 0
                    # Store only unsuppressed boxes for this class
                    class_boxes = class_decoded_locs[1 - suppress]
                    # set label index to class c + 1
                    # class_labels = torch.LongTensor((1 - suppress).sum().item() * [c + 1]).to(device)
                    class_scores = class_scores[1 - suppress]
                else:
                    # If no object in any class is found, store a placeholder for 'background'
                    class_boxes = torch.FloatTensor([]).to(device)
                    # class_labels = torch.LongTensor([]).to(device)
                    class_scores = torch.FloatTensor([]).to(device)
                
                n_objects = class_boxes.shape[0]
                # Keep only the top k objects
                if n_objects > top_k:
                    class_boxes = class_boxes[:top_k]  # (top_k, 4)
                    # class_labels = class_labels[:top_k]  # (top_k)
                    class_scores = class_scores[:top_k]  # (top_k)

                image_boxes.append(class_boxes)
                # image_labels.append(class_labels)
                image_scores.append(class_scores)
                #-----------------------------------------------------------------------------------

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            # all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_scores  # lists of length batch_size ---- all_images_labels, 


class MultiBoxLossFork(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLossFork, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, n_classes, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, n_classes, 8732, 2)
        :param boxes: true  object bounding boxes in boundary coordinates, a tensor of dimensions (N, n_classes, x, 4)
        :param labels: true object labels, a tensor of dimensions (N, n_classes, x, 1)
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(1)

        assert n_priors == predicted_locs.size(2) == predicted_scores.size(2)

        true_locs = torch.zeros((batch_size, n_classes, n_priors, 4), dtype=torch.float).to(device)  # (N, n_classes, 8732, 4)
        true_classes = torch.zeros((batch_size, n_classes, n_priors), dtype=torch.long).to(device)  # (N, n_classes, 8732)

        # For each image
        for i in range(batch_size):
            for c in range(n_classes):
                n_objects = boxes[i][c].size(0)
                # without any type of box on the image
                if n_objects == 0:
                    continue

                overlap = find_jaccard_overlap(boxes[i][c], self.priors_xy)  # (n_objects, 8732)

                # 找每個prior和哪個positive box有最大重疊
                # For each prior, find the object that has the maximum overlap
                # overlap_for_each_prior    : max value
                # object_for_each_prior     : max value index(positive box index)
                overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732) 0~n_objects-1

                # We don't want a situation where an object is not represented in our positive (non-background) priors -
                # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
                # 2. All priors with the object may be assigned as background based on the threshold (0.5).

                # To remedy this -
                # First, find the prior that has the maximum overlap for each object.
                _, prior_for_each_object = overlap.max(dim=1)  # (n_objects) 0~8731

                # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
                object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

                # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
                overlap_for_each_prior[prior_for_each_object] = 1.

                # Labels for each prior
                label_for_each_prior = labels[i][c][object_for_each_prior]  # (8732)
                # Set priors whose overlaps with objects are less than the threshold to be background (no object)
                label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

                # Store
                true_classes[i][c] = label_for_each_prior

                # Encode center-size object coordinates into the form we regressed predicted boxes to
                true_locs[i][c] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][c][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # exclude the background class
        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, n_classes, 8732)

        # todo : make it better to work
        # turn batch first to n_classes first
        true_locs = true_locs.permute(1, 0, 2, 3)
        true_classes = true_classes.permute(1, 0, 2)
        positive_priors = positive_priors.permute(1, 0, 2)

        cp_predicted_locs = predicted_locs.permute(1, 0, 2, 3)
        cp_predicted_scores = predicted_scores.permute(1, 0, 2, 3)

        # LOCALIZATION LOSS

        # fork set default weight [0.2, 0.2, 0.4, 0.2] for frame, text, face, body seperately.
        # we use uniform weight for testing
        class_weight = n_classes * [1 / n_classes]

        ret_loss = 0
        for c in range(n_classes):
            class_pos_prior = positive_priors[c, :, :] # (N, 8732)

            # mean there is no positive
            if class_pos_prior.sum() == 0:
                continue

            # Localization loss is computed only over positive (non-background) priors
            loc_loss = self.smooth_l1(cp_predicted_locs[c][class_pos_prior], true_locs[c][class_pos_prior])  # (), scalar

            # CONFIDENCE LOSS

            # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
            # That is, FOR EACH IMAGE,
            # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
            # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

            # Number of positive and hard-negative priors per image
            n_positives = class_pos_prior.sum(dim=1)  # (N)
            n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

            # First, find the loss for all priors
            conf_loss_all = self.cross_entropy(cp_predicted_scores[c].view(-1, 2), true_classes[c].view(-1))  # (N * 8732)
            conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

            # We already know which priors are positive
            conf_loss_pos = conf_loss_all[class_pos_prior]  # (sum(n_positives))

            # Next, find which priors are hard-negative
            # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
            conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
            conf_loss_neg[class_pos_prior] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
            conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
            hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
            hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
            conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

            # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
            # conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar
            conf_loss = conf_loss_hard_neg.sum() + conf_loss_pos.sum()
            # print(loc_loss, conf_loss)

            # TOTAL LOSS
            # ret_loss = ret_loss + class_weight[c] * (conf_loss + self.alpha * loc_loss) 
            ret_loss = ret_loss + class_weight[c] * (conf_loss + self.alpha * loc_loss) / n_positives.sum().float()

        return ret_loss


def build_fork_model_and_loss_function(n_classes):
    model = SSD300Fork(n_classes=n_classes)
    loss_func = MultiBoxLossFork(priors_cxcy=model.priors_cxcy)
    return model, loss_func

# test for using fork model
if __name__ == '__main__':
    import os
    import torch
    from PIL import Image
    import torchvision.transforms.functional as TF

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = "v2-e82cb9a688c9cba9265e1044c5159d7b_hd.jpg"
    img = Image.open(path, mode='r').convert('RGB')
    img = img.resize((300, 300))
    img = TF.to_tensor(img)

    imgs = torch.stack([img], dim=0)
    imgs = imgs.to(device)

    boxes = [
        [[0, 0, 10, 10], 
        [30, 30, 50, 50]],
        [[70, 70, 100, 100], 
        [100, 100, 120, 120],
        [65, 65, 120, 120]]
    ]
    labels = [
        [1, 1], 
        [1, 1, 1]
    ]

    fake_batch = 1
    fake_class = 2
    boxes = [boxes] * fake_batch
    labels = [labels] * fake_batch

    for i in range(fake_batch):
        for j in range(fake_class):
            boxes[i][j] = torch.FloatTensor(boxes[i][j]).to(device)
            labels[i][j] = torch.FloatTensor(labels[i][j]).to(device)

    model = SSD300Fork(2)
    model.eval()
    model.to(device)
    # model = SSD300()
    # print(model)

    loss_func = MultiBoxLossFork(priors_cxcy=model.priors_cxcy)
    loss_func.to(device)

    predicted_locs, predicted_scores = model(imgs)
    print(predicted_locs.shape, predicted_scores.shape)

    print(loss_func(predicted_locs, predicted_scores, boxes, labels))

    # # threshold = 0.5
    # min_score = 0.5
    # max_overlap = 0.5
    # top_k = 200

    # det_boxes, det_labels, det_scores = model.detect_objects(
    #     predicted_locs, 
    #     predicted_scores, 
    #     min_score=min_score,
    #     max_overlap=max_overlap, 
    #     top_k=top_k)

    # print(det_boxes, det_labels, det_scores)