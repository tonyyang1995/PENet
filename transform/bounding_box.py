import numbers
import numpy as np
import torch

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

def _validate_bboxes(bboxes):
    if isinstance(bboxes, tuple) or isinstance(bboxes, np.ndarray):
        bboxes = torch.tensor(bboxes).float()
    elif isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.clone().detach().float()
    else:
        raise ValueError("Type of bboxes should be `tuple`, `numpy.ndarray` or `torch.Tensor`. Got {}".format(type(bboxes)))

    if bboxes.ndimension() != 2:
        raise ValueError("Dimensions of bbox should be 2. Got {}".format(bboxes.ndimension()))
    if bboxes.size(0) == 0:
        raise ValueError("There should be at least one bounding box. Got {}".format(bboxes.size(0)))
    if bboxes.size(-1) != 8 and bboxes.size(-1) != 6 :
        raise ValueError("Last dimenion of bboxes should be 8 or 6 (including classes). Got {}".format(bboxes.size(-1)))

    return bboxes

def _validate_size(size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))

    return size

class BBox(object):
    def __init__(self, bboxes, image_size):

        self.size = image_size
        self.bboxes = bboxes

    def bbox_sizes(self):
        return self.bboxes.size()

    def add_gt(self, left, top, width, height, cls_id, score=1.0):
        # if the sample cover some ground truth, remove the ground truth

        addition_gt = torch.Tensor((cls_id, left, top, width, height)).float().view(1,-1)
        self.bboxes = torch.cat((self.bboxes, addition_gt))

    @staticmethod
    def from_xyxy(bboxes, image_size):
        bboxes = _validate_bboxes(bboxes)
        image_size = _validate_size(image_size)
        
        return BBox(bboxes, image_size)

    @staticmethod
    def from_xyhw(bboxes, image_size, normalized=False):
        bboxes = _validate_bboxes(bboxes)
        image_size = _validate_size(image_size)
        
        w_factor, h_factor = image_size if normalized else (1, 1)
        classes, tx, ty, w, h = bboxes.split(1, dim=-1)
        xmin = w_factor * (tx - w/2)
        ymin = h_factor * (ty - h/2)
        xmax = w_factor * (tx + w/2)
        ymax = h_factor * (ty + h/2)
        
        return BBox(torch.cat((classes, xmin, ymin, xmax, ymax), dim=-1), image_size)

    @staticmethod
    def from_yolo(bboxes, image_size):
        return BBox.from_xyhw(bboxes, image_size, normalized=True)

    @staticmethod
    def from_visDrone(bboxes, image_size, normalized=False):
        #w_factor, h_factor = image_size if normalized else (1, 1)
        bboxes = _validate_bboxes(bboxes)
        image_size = _validate_size(image_size)
        #print(image_size)
        box_left, box_top, box_width, box_height, score, classes, trunc, occlusion = bboxes.split(1, dim=-1)
        xmin = box_left
        ymin = box_top
        xmax = (box_left + box_width)
        ymax = (box_top + box_height)
        #print(xmin[0], ymin[0], xmax[0], ymax[0])
        return BBox(torch.cat((classes, xmin, ymin, xmax, ymax), dim=-1), image_size)

    @staticmethod
    def from_uavDT(bboxes, image_size, normalized=False):
        bboxes = _validate_bboxes(bboxes)
        image_size = _validate_size(image_size)

        box_left, box_top, box_width, box_height, score, classes = bboxes.split(1, dim=-1)
        xmin = box_left
        ymin = box_top
        xmax = (box_left + box_width)
        ymax = (box_top + box_height)
        return BBox(torch.cat((classes, xmin, ymin, xmax, ymax), dim=-1), image_size)

    @staticmethod
    def from_uavDT_coarse(bboxes, image_size, normalized=False):
        bboxes = _validate_bboxes(bboxes)
        image_size = _validate_size(image_size)

        box_left, box_top, box_width, box_height, score, classes = bboxes.split(1, dim=-1)
        xmin = box_left
        ymin = box_top
        xmax = (box_left + box_width)
        ymax = (box_top + box_height)
        classes[classes < 10] = 1
        return BBox(torch.cat((classes, xmin, ymin, xmax, ymax), dim=-1), image_size)

    @staticmethod
    def from_visDrone_coarse(bboxes, image_size, normalized=False):
        #w_factor, h_factor = image_size if normalized else (1, 1)
        bboxes = _validate_bboxes(bboxes)
        image_size = _validate_size(image_size)
        #print(image_size)
        box_left, box_top, box_width, box_height, score, classes, trunc, occlusion = bboxes.split(1, dim=-1)
        xmin = box_left
        ymin = box_top
        xmax = (box_left + box_width)
        ymax = (box_top + box_height)
        classes[classes == 11.0] = 0
        classes[classes > 0] = 1
        #print(classes)
        #print(xmin[0], ymin[0], xmax[0], ymax[0])
        return BBox(torch.cat((classes, xmin, ymin, xmax, ymax), dim=-1), image_size)

    def _split(self, mode='xyxy'):
        if mode == 'xyxy':
            return self.bboxes.clone().split(1, dim=-1)
        elif mode == 'xyhw':
            classes, xmin, ymin, xmax, ymax = bboxes.split(1, dim=-1)
            tx = xmin + xmax
            ty = ymin + ymax
            w = xmax - xmin
            h = ymax - ymin
            return classes, tx, ty, w, h

    def to_yolo(self):
        w_factor, h_factor = self.size
        classes, xmin, ymin, xmax, ymax = self.bboxes.clone()._split()
        tx = (xmin + xmax) / (2*w_factor)
        ty = (ymin + ymax) / (2*h_factor)
        w = (xmax - xmin) / w_factor
        h = (ymax - ymin) / h_factor
        
        return torch.cat((classes, tx, ty, w, h), -1)

    def to_tensor(self, mode='yolo'):
        if mode not in ('yolo', 'xyhw', 'xyxy'):
            raise ValueError("BBox only supports mode: `yolo`, `xyhw`, `xyxy`. Got {}".format(mode))

        if mode == 'xyxy':
            #return self.bboxes.clone()
            bboxes = self.bboxes.clone()
            classes, xmin, ymin, xmax, ymax = self._split()
            score = torch.ones(classes.size(0), 1)
            return torch.cat((xmin, ymin, xmax, ymax, score, classes), -1)
        else:
            w_factor, h_factor = self.size if mode == 'yolo' else (1, 1)
            classes, xmin, ymin, xmax, ymax = self._split()
            tx = (xmin + xmax) / (2*w_factor)
            ty = (ymin + ymax) / (2*h_factor)
            w = (xmax - xmin) / w_factor
            h = (ymax - ymin) / h_factor
        
            return torch.cat((classes, tx, ty, w, h), -1)

    def to_yolo(self):
        return self.to_tensor(mdde='yolo')        


    def crop(self, box):
        """Crop the bboxes.
        Args:
            box: 4-tuple
        """
        w, h = box[2] - box[0], box[3] - box[1]
        classes, xmin, ymin, xmax, ymax = self._split()
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        cropped_bboxes = torch.cat(
            (classes, cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1)
        
        is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)
        is_empty = is_empty.view(-1)
        cropped_bboxes = cropped_bboxes[is_empty == 0]

        return BBox(cropped_bboxes, (w, h))

    def resize(self, box_size):
        w, h = self.size
        box_w, box_h = _validate_size(box_size)
        ratio_w, ratio_h = box_w / w, box_h / h
        
        classes, xmin, ymin, xmax, ymax = self._split()
        resized_xmin = (xmin - box_w).clamp(min=0, max=w)
        resized_ymin = (ymin - box_h).clamp(min=0, max=h)
        resized_xmax = (xmax - box_w).clamp(min=0, max=w)
        resized_ymax = (ymax - box_h).clamp(min=0, max=h)

        resized_bboxes = torch.cat(
            (classes, resized_xmin, resized_ymin, resized_xmax, resized_ymax), dim=-1)

        return BBox(resized_bboxes, self.size)
        
            
    def crop_and_zoom_out(self, box, zoom_out):
        zoom_out = _validate_size(zoom_out)
        
        w, h = box[2] - box[0], box[3] - box[1]
        zw, zh = zoom_out
        
        classes, xmin, ymin, xmax, ymax = self._split()
        box_w = xmax - xmin
        box_h = ymax - ymin
        xmin[box_w < zw] = xmin[box_w < zw] - zw/2
        ymin[box_h < zh] = ymin[box_h < zh] - zh/2
        xmax[box_w < zw] = xmax[box_w < zw] + zw/2
        ymax[box_h < zh] = ymax[box_h < zh] + zh/2

        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        cropped_bboxes = torch.cat(
            (classes, cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1)
        
        # remove bounding boxes out of the crop region
        is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)
        is_empty = is_empty.view(-1)
        cropped_bboxes = cropped_bboxes[is_empty == 0]

        return BBox(cropped_bboxes, (w, h))

    
    def pad(self, padding):
        left, right, top, down = padding
        classes, xmin, ymin, xmax, ymax = self._split()
        xmin += left
        ymin += top
        xmax += right
        ymax += down

        padded_bboxes = torch.cat((classes, xmin, ymin, xmax, ymax), -1)
        w, h = self.size
        padded_w = w + left + right
        padded_h = h + top + down

        return BBox(padded_bboxes, (padded_w, padded_h))

    def rotate(self, angle):
        w, h = self.size
        classes, xmin, ymin, xmax, ymax = self._split()
        tx = w / 2 # center of image
        ty = h / 2
        
        rotated_xmin = torch.cos(angle) * (xmin-tx) - torch.sin(angle) * (ymin - ty) + tx
        rotated_ymin = torch.sin(angle) * (xmin-tx) + torch.cos(angle) * (ymin - ty) + ty
        rotated_xmax = torch.cos(angle) * (xmax-tx) - torch.sin(angle) * (ymax - ty) + tx
        rotated_ymax = torch.sin(angle) * (xmax-tx) + torch.cos(angle) * (ymax - ty) + ty

        rotated_bboxes = torch.cat(
            (classes, rotated_xmin, rotated_ymin, rotated_xmax, rotated_ymax), -1)

    def hflip(self):
        w, h = self.size
        classes, xmin, ymin, xmax, ymax = self._split()
        transposed_xmin = w - xmax
        transposed_xmax = w - xmin - 1

        transposed_bboxes = torch.cat(
                (classes, transposed_xmin, ymin, transposed_xmax, ymax), dim=-1)

        return BBox(transposed_bboxes, (w, h))

    def vflip(self):
        w, h = self.size
        classes, xmin, ymin, xmax, ymax = self._split()
        transposed_ymin = h - ymax
        transposed_ymax = h - ymin

        transposed_bboxes = torch.cat(
                (classes, xmin, transposed_ymin, xmax, transposed_ymax), dim=-1)
            
        return BBox(transposed_bboxes, (w, h))

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError("Only horizontal and vertical flipping are supported")
        
        if method == FLIP_LEFT_RIGHT:
            return self.hflip()
        else:
            return self.vflip()

    @staticmethod
    def iou(bbox1, bbox2):
        _, b1_x1, b1_y1, b1_x2, b1_y2 = bbox1
        _, b2_x1, b2_y1, b2_x2, b2_y2 = bbox2

        if (b1_x1 < b2_x1 and b1_x2 > b2_x2 and b1_y1 < b2_y1 and b1_y2 > b2_y2):
            return 1.0
        if (b2_x1 < b1_x1 and b2_x2 > b1_x2 and b2_y1 < b1_y1 and b2_y2 > b1_y2):
            return 1.0

        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=1) * \
            torch.clamp(inter_rect_y2 - inter_rect_y1, min=1)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou

    def non_max_merge(self, box_size=1080, iou_thresh=0.2):
        w, h = self.size
        box_w, box_h = _validate_size(box_size)

        bboxes = self.bboxes.clone()

        bboxes_to_merge = []
        i = 0
        visited = set()
        for i in range(bboxes.size(0)):
            if i in visited:
                continue
            
            visited.add(i)
            bbox_i = bboxes[i, :]
            tx = (bbox_i[1] + bbox_i[3]) / 2
            ty = (bbox_i[2] + bbox_i[4]) / 2
            
            if tx - box_w/2 < 0:
                bbox_i[1] = 0
                bbox_i[3] = box_w
            elif tx + box_w/2 > w:
                bbox_i[3] = w
                bbox_i[1] = w - box_w
            else:
                bbox_i[1] = tx - box_w/2
                bbox_i[3] = tx + box_w/2

            if ty - box_h/2 < 0:
                bbox_i[2] = 0
                bbox_i[4] = box_h
            elif ty + box_h/2 > h:
                bbox_i[4] = h
                bbox_i[2] = h - box_h
            else:
                bbox_i[2] = ty - box_h/2
                bbox_i[4] = ty + box_h/2

            bboxes_to_merge.append(bbox_i)
            for j in range(i+1, bboxes.size(0)):
                bbox_j = bboxes[j, :]
                if bbox_i[0] != bbox_j[0]: # bboxes with different labels are not merged
                    continue
                if BBox.iou(bbox_i, bbox_j) > iou_thresh:
                    visited.add(j)
        return BBox(torch.stack(bboxes_to_merge, 0), (w, h))