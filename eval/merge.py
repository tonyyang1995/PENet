from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os
import glob
import json

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import matplotlib.patches as patches

np.set_printoptions(precision=3, suppress=True)

plt.rcParams["figure.figsize"] = (20,20)

def iou(bbox1, bbox2):
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2

    if (b1_x1 < b2_x1 and b1_x2 > b2_x2 and b1_y1 < b2_y1 and b1_y2 > b2_y2):
        return 1.0

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, a_min=1, a_max=None) * \
        np.clip(inter_rect_y2 - inter_rect_y1, a_min=1, a_max=None)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    # print(iou)
    return iou

def inter_area(bbox1, bbox2):
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2

    if (b1_x1 < b2_x1 and b1_x2 > b2_x2 and b1_y1 < b2_y1 and b1_y2 > b2_y2):
        return 1.0

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    area = np.clip(inter_rect_x2 - inter_rect_x1, a_min=1, a_max=None) * \
        np.clip(inter_rect_y2 - inter_rect_y1, a_min=1, a_max=None)

    return area

def contain(bbox1, bbox2):
    area1 = (bbox1[3] - bbox1[1]) * (bbox1[2] - bbox1[0])
    area2 = (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])
    if area2 > area1:
        temp = bbox1
        bbox1 = bbox2
        bbox2 = temp
    if bbox1[0] < bbox2[0] and bbox1[1] < bbox2[1] and bbox1[2] > bbox2[0] and bbox1[3] > bbox2[3]:
        return True
    
    if inter_area(bbox1, bbox2) / area2 > 0.8:
        return True
    return False

def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def merge(bbox1, bbox2):
    score = bbox1[5] if bbox1[5] < bbox2[5] else bbox2[5]
    x1 = bbox1[0] if bbox1[0] < bbox2[0] else bbox2[0]
    y1 = bbox1[1] if bbox1[1] < bbox2[1] else bbox2[1]
    x2 = bbox1[2] if bbox1[2] > bbox2[2] else bbox2[2]
    y2 = bbox1[3] if bbox1[3] > bbox2[3] else bbox2[3]
    return np.array([x1, y1, x2, y2, bbox1[4], score])


def adjacent_h(bbox1, bbox2):
    height_diff = abs(bbox1[3] - bbox1[1] - bbox2[3] + bbox2[1])
    if height_diff < 5 and dist(bbox1[2], bbox1[1], bbox2[0], bbox2[1]) < 10:
        return True
    
    return False


# def adjacent_h(bbox1, bbox2):
#     if bbox1[0] > bbox2[0]:
#         temp = bbox1
#         bbox1 = bbox2
#         bbox2 = temp
#     height_diff = abs(bbox1[3] - bbox1[1] - bbox2[3] + bbox2[1])
#     if height_diff < 5 and dist(bbox1[2], bbox1[3], bbox2[0], bbox2[3]) < 10:
#         return True
    
#     return False


def adjacent_w(bbox1, bbox2):
    width_diff = abs(bbox1[2] - bbox1[0] - bbox2[2] + bbox2[0])
    if width_diff < 5 and dist(bbox1[0], bbox1[3], bbox2[0], bbox2[1]) < 10:
        return True

    return False

# def adjacent_w(bbox1, bbox2):
#     if bbox1[1] > bbox2[1]:
#         temp = bbox1
#         bbox1 = bbox2
#         bbox2 = bbox1
#     width_diff = abs(bbox1[2] - bbox1[0] - bbox2[2] + bbox2[0])
#     if width_diff < 5 and dist(bbox1[0], bbox1[1], bbox2[0], bbox2[3]) < 10:
#         return True

#     return False
    

if __name__ == "__main__":
    # image_dir ='visDrone/images'
    # annotation_dir = 'visDrone/valdetection'#'annotations'
    # merged_annotation_dir = 'visDrone/merged_valdetection'
    image_dir ='uavdt/images'
    annotation_dir = 'uavdt/valdetection'#'annotations'
    merged_annotation_dir = 'uavdt/merged_valdetection'

    os.makedirs(merged_annotation_dir, exist_ok=True)

    fig, ax = plt.subplots(1)
    for image_path in glob.glob(os.path.join(image_dir, '*')):
        image_name = os.path.basename(image_path)
        # if '0000117_00112_d_0000087' not in image_name:
        #     continue
        print(image_name)
        image = imread(image_path)

        annotation_path = os.path.join(annotation_dir, image_name.replace('jpg', 'txt'))
        # bboxes = np.loadtxt(annotation_path, delimiter=',')
        bboxes = np.loadtxt(annotation_path, dtype='float', delimiter=' ')
        merged_annotation_path = os.path.join(merged_annotation_dir, image_name.replace('jpg', 'txt'))

        if len(bboxes.shape) == 1:
            f = open(merged_annotation_path, 'w')
            continue

        
        # bboxes = bboxes[bboxes[:, 4] == 4]
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        bboxes = bboxes[np.argsort(areas)[::-1]]
        # print(bboxes)

        visited = set()
        keeps = []
        i = 0
        while i < bboxes.shape[0]:
            bbox_i = bboxes[i, ...]
            # if bboxes[i, 4] != 4:
            #     keeps.append(bbox_i)
            #     i = i + 1
            #     continue

            if i in visited:
                i = i + 1
                continue
            
            for j in range(i+1, bboxes.shape[0]):
                if j in visited or bbox_i[4] != bboxes[j, 4]: # categories not equal
                    continue
                
                bbox_j = bboxes[j, ...]
                if contain(bbox_i[:4], bbox_j[:4]):
                    visited.add(j)
                    break
                # if adjacent_h(bbox_i, bbox_j) or adjacent_w(bbox_i, bbox_j):
                #     merged_bbox = merge(bbox_i, bbox_j)
                #     bboxes[i, ...] = merged_bbox
                #     visited.add(j)
                #     break
            
            if j >= bboxes.shape[0]-1:
                keeps.append(bbox_i)
                i = i + 1

        #if len(keeps) != 0:
        np.savetxt(merged_annotation_path, np.stack(keeps), fmt='%f')
        
        # ax.set_axis_off()
        # ax.imshow(image)
        
        # for bbox in keeps:
        #     if bbox[4] == 0:
        #         continue
        #     bbox[2] = bbox[2] - bbox[0]
        #     bbox[3] = bbox[3] - bbox[1]

        #     category_id = int(bbox[4])
        #     rect = patches.Rectangle(bbox[:2], width=bbox[2], height=bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        #     ax.text(
        #         bbox[0], bbox[1], s = f'{category_id}', color = 'white', 
        #         verticalalignment = 'top', bbox = {'color': 'black', 'pad': 0}
        #     )

        # fig.savefig(f'val/{image_name}', bbox_inches='tight', pad_inches=0)