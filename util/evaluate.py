from __future__ import division
import math
import time
import tqdm

import numpy as np
import torch

def soft_nms(dets, sigma=0.5, thresh=0.001, cuda=1):
    #print(dets.shape)
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N,1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N,1)

    dets = torch.cat((dets, indexes), dim=1)

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        tscore = scores[i].clone()
        pos = i + 1

        if i != N-1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        xx1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        yy1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].long()

    return dets[keep]


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area  

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.type(torch.DoubleTensor), b2_x1.type(torch.DoubleTensor))
    inter_rect_y1 = torch.max(b1_y1.type(torch.DoubleTensor), b2_y1.type(torch.DoubleTensor))
    inter_rect_x2 = torch.min(b1_x2.type(torch.DoubleTensor), b2_x2.type(torch.DoubleTensor))
    inter_rect_y2 = torch.min(b1_y2.type(torch.DoubleTensor), b2_y2.type(torch.DoubleTensor))
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area.type(torch.DoubleTensor) / (b1_area.type(torch.DoubleTensor) + b2_area.type(torch.DoubleTensor) - inter_area.type(torch.DoubleTensor) + 1e-16)

    return iou

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp: True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall: The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_batch_statistics(outputs, targets, iou_threshold=0.5):

    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]

        pred_boxes = output[:, :4]
        pred_scores = output[:, -1]
        pred_labels = output[:, 4]

        true_positives = np.zeros(pred_boxes.shape[0])

        #annotations = targets[targets[:, 0] == sample_i]
        target_labels = targets[:, 5] if len(targets) else []

        if len(targets):
            detected_boxes = []
            target_boxes = targets[:, :4]
            target_boxes[:, 2] = target_boxes[:, 0] + target_boxes[:, 2]
            target_boxes[:, 3] = target_boxes[:, 1] + target_boxes[:, 3]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                # If targets are found break
                if len(detected_boxes) == len(targets):
                    break
                # Ignore if label is not one of the target labels
                if pred_label.double() not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes.float()).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]

        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def nms(predictions, threshold=0.5):
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]

    scores = predictions[:, 4]
    #print(scores.shape)
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]

    keep = []
    while(order.size > 0):
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #print(ovr)
        inds = np.where(ovr <= threshold)[0]

        order = order[inds + 1]

    return predictions[keep]

def rid(predictions, threshold=0.8):
    ox1 = predictions[:, 0]
    oy1 = predictions[:, 1]
    ox2 = predictions[:, 2]
    oy2 = predictions[:, 3]
    oscores = predictions[:, 4]
    clses = predictions[:, 5]
    
    keeps = []
    for c in range(1, 11):
        index = (clses == c)
        x1 = ox1[index]
        y1 = oy1[index]
        x2 = ox2[index]
        y2 = oy2[index]
        scores = oscores[index]

        areas = (x2 - x1) * (y2 -y1)
        order = scores.argsort()[::-1]

        keep = []
        while (order.size > 0):
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h

            ia = inter / areas[order[1:]]
            inds = np.where(ia <= threshold)[0]
            order = order[inds + 1]

        keeps += keep
    return predictions[keeps]


def post_processing(predictions, threshold):
    ox1 = predictions[:, 0]
    oy1 = predictions[:, 1]
    ox2 = predictions[:, 2]
    oy2 = predictions[:, 3]
    oscores = predictions[:, 4]
    clses = predictions[:, 5]

    keeps = []
    for c in range(1, 11):
      index = (clses == c)
      x1 = ox1[index]
      y1 = oy1[index]
      x2 = ox2[index]
      y2 = oy2[index]
      scores = oscores[index]

      areas = (x2 - x1) * (y2 -y1)
      order = scores.argsort()[::-1]
      #order = areas.argsort()[::-1]
      keep = []
      while (order.size > 0):
          i = order[0]
          keep.append(i)
          xx1 = np.maximum(x1[i], x1[order[1:]])
          yy1 = np.maximum(y1[i], y1[order[1:]])
          xx2 = np.minimum(x2[i], x2[order[1:]])
          yy2 = np.minimum(y2[i], y2[order[1:]])

          w = np.maximum(0.0, xx2 - xx1 + 1)
          h = np.maximum(0.0, yy2 - yy1 + 1)

          inter = w * h

          # ia = inter / areas[i]
          #print(inter)
          #print(areas[i])
          #print(areas[order[1:]])
          ovr = inter / (areas[i] + areas[order[1:]] - inter)
          #print(ovr)
          #assert False
          inds = np.where(ovr <= threshold)[0]

          order = order[inds + 1]
          #assert False

      keeps += keep
    #return np.stack(keeps)#
    return predictions[keeps]

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    #prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    #print(prediction.size())

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if image_pred.size(0) == 0:
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output