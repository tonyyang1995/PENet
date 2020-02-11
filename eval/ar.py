from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os
import json
import shutil

from PIL import Image
#import skimage.io as io
import matplotlib.pyplot as plt


def int_filename(gt_path, dt_path, gt_image_path, gt_int_path, dt_int_path, gt_image_int_path):
    os.makedirs(gt_int_path, exist_ok=True)
    os.makedirs(dt_int_path, exist_ok=True)
    os.makedirs(gt_image_int_path, exist_ok=True)

    file_ids = []
    for root, dirs, files in os.walk(gt_path):
        for filename in files:
            file_ids.append(filename)

    sorted(file_ids)
    for idx, file_id in enumerate(file_ids):
        print(file_id)
        gt_filepath = os.path.join(gt_path, file_id)
        dt_filepath = os.path.join(dt_path, file_id)
        gt_image_filepath = os.path.join(gt_image_path, f'{file_id.split(".")[0]}.jpg')
        
        gt_int_filename = os.path.join(gt_int_path, f'{idx}.txt')
        dt_int_filename = os.path.join(dt_int_path, f'{idx}.txt')
        gt_image_int_filename = os.path.join(gt_image_int_path, f'{idx}.jpg')
        shutil.copy(gt_filepath, gt_int_filename)
        shutil.copy(dt_filepath, dt_int_filename)
        shutil.copy(gt_image_filepath, gt_image_int_filename)


def txt_to_coco_gt(src_path, image_path, dst_filepath):
    visDrone_coco =	{
        "info":{"description": "visDrone 2019 Dataset","url" : "unknown","year" : 2019 ,"contributor" : "unknown" ,"date_created" : "unknown"},
        "license": [{"url" : "unknown","id" : 1,"name": "unknown"}],
        "images":[],
        "annotations":[],
        "categories":[
            #{"supercategory" : "ignored regions", "id" : 0, "name": "ignored regions"},
            {"supercategory" : "pedestrian", "id" : 1, "name" : "pedestrian"},
            {"supercategory" : "people", "id" : 2, "name" : "people"},
            {"supercategory" : "bicycle", "id" : 3, "name" : "bicycle"},
            {"supercategory" : "car", "id" : 4, "name" : "car"},
            {"supercategory" : "van", "id" : 5, "name" : "van"},
            {"supercategory" : "truck", "id" : 6, "name" : "truck"},
            {"supercategory" : "tricycle", "id" : 7, "name" : "tricycle"},
            {"supercategory" : "awning tricycle", "id" : 8, "name" : "awning tricycle"},
            {"supercategory" : "bus", "id" : 9, "name" : "bus"},
            {"supercategory" : "motor", "id" : 10, "name" : "motor"}]
            #{"supercategory" : "others", "id" : 11, "name" : "others"}]
    }
    anno_id = 1

    for root, dirs, files in os.walk(src_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_id = int(filename.split('.')[0])
            print(filepath)
            image_filepath = os.path.join(image_path, f'{file_id}.jpg')
            im = Image.open(image_filepath).convert('RGB')



            visDrone_coco['images'].append({
                "licenses":1, "file_name": image_filepath, "coco_url":"null", "height": im.size[1],"width": im.size[0], "data_captured": "nunknown", "flicker_url": "unknown", "id": file_id
            })

            with open(filepath, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    bbox = line.split(',')
                    bbox_coord = bbox[:4]
                    visDrone_coco['annotations'].append({
                        "segmentation":[], "area":float(bbox_coord[2])*float(bbox_coord[3]), "iscrowd": 0, "image_id": file_id, "bbox":[int(i) for i in bbox_coord], "category_id": int(bbox[5]), "id": anno_id
                    })
                    anno_id += 1
                
            
        with open(dst_filepath, 'w') as jf:
            json.dump(visDrone_coco, jf)
            

def txt_to_coco_gt2(src_path, image_path, dst_filepath):
    uavdt_coco = {
        "info":{"description": "uavdt 2019 Dataset","url" : "unknown","year" : 2019 ,"contributor" : "unknown" ,"date_created" : "unknown"},
        "license": [{"url" : "unknown","id" : 1,"name": "unknown"}],
        "images":[],
        "annotations":[],
        "categories":[
            {"supercategory" : "car", "id" : 1, "name" : "awning tricycle"},
            {"supercategory" : "bus", "id" : 2, "name" : "bus"},
            {"supercategory" : "truck", "id" : 3, "name" : "motor"}]
    }
    anno_id = 1

    for root, dirs, files in os.walk(src_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_id = int(filename.split('.')[0])
            print(filepath)
            image_filepath = os.path.join(image_path, f'{file_id}.jpg')
            im = Image.open(image_filepath).convert('RGB')



            uavdt_coco['images'].append({
                "licenses":1, "file_name": image_filepath, "coco_url":"null", "height": im.size[1],"width": im.size[0], "data_captured": "nunknown", "flicker_url": "unknown", "id": file_id
            })

            with open(filepath, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    bbox = line.split(',')
                    bbox_coord = bbox[:4]
                    uavdt_coco['annotations'].append({
                        "segmentation":[], "area":float(bbox_coord[2])*float(bbox_coord[3]), "iscrowd": 0, "image_id": file_id, "bbox":[int(i) for i in bbox_coord], "category_id": int(bbox[5]), "id": anno_id
                    })
                    anno_id += 1
                
            
        with open(dst_filepath, 'w') as jf:
            json.dump(uavdt_coco, jf)
                
def txt_to_coco_dt(src_path, dst_filepath):

    bboxes = []
    for root, dirs, files in os.walk(src_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_id = filename.split('.')[0]
            print(filepath)
        
            with open(filepath, 'r') as f:
                lines = f.readlines()
                #print(len(lines))
                for line in lines:
                    bbox = line.rstrip('\n').split(' ')
                    #bbox_coord = bbox[:4]
                    bbox[2] = float(bbox[2]) - float(bbox[0])
                    bbox[3] = float(bbox[3]) - float(bbox[1])
                    bbox_coord = bbox[:4]

                    bbox_dict = {
                        'image_id': int(file_id),
                        'category_id': int(float(bbox[4])),
                        'score': float(bbox[5]),
                        'bbox': [float(i) for i in bbox_coord]
                    }
                    bboxes.append(bbox_dict)
                
            
        with open(dst_filepath, 'w') as jf:
            json.dump(bboxes, jf)


def txt_to_coco_dt2(src_path, dst_filepath):

    bboxes = []
    for root, dirs, files in os.walk(src_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_id = filename.split('.')[0]
            print(filepath)
        
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    bbox = line.split(',')
                    bbox_coord = bbox[:4]
                    bbox_dict = {
                        'image_id': int(file_id),
                        'category_id': int(bbox[5]),
                        'score': float(bbox[4]),
                        'bbox': [int(i) for i in bbox_coord]
                    }
                    bboxes.append(bbox_dict)
                
            
        with open(dst_filepath, 'w') as jf:
            json.dump(bboxes, jf)

def evaluate(gt_path, dt_path, ann_type='bbox'):
    file_ids = []
    for root, dirs, files in os.walk(gt_path):
        for filename in files:
            file_id = filename.split('.')[0]
            print(file_id)
            file_ids.append(int(file_id))

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(dt_path)

    imgIds = sorted(coco_gt.getImgIds())
    print(len(imgIds))

    # image_ids = coco_gt.getImgIds(imgIds = [72])
    # img = coco_gt.loadImgs(image_ids)[0]
    # I = io.imread(img['file_name'])
    # plt.axis('off')
    # plt.imshow(I)

    # annIds = coco_gt.getAnnIds(imgIds=img['id'], iscrowd=None)
    # anns = coco_gt.loadAnns(annIds)
    # print(anns)
    # coco_gt.showAnns(anns)


    cocoEval = COCOeval(coco_gt, coco_dt, ann_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [1,10,100,500]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    
if __name__ == "__main__":
    # gt_path, gt_image_path, dt_path = 'annotations', 'images','../results/stack/valdetection'
    dataset = 'uavdt'
    #dataset = 'visDrone'
    gt_path, gt_image_path, dt_path = dataset+'/annotations', dataset+'/images',dataset+'/merged_valdetection'

    gt_int_path, gt_image_int_path, dt_int_path = dataset+'/annotations_int', dataset+'/images_int',dataset+'/valdetection_int'
    gt_json_path, dt_json_path = dataset+'/annotations.json', dataset+'/valdetection_fine_ori.json'
    
    int_filename(gt_path, dt_path, gt_image_path, gt_int_path, dt_int_path, gt_image_int_path)
    txt_to_coco_gt2(gt_int_path, gt_image_int_path, gt_json_path)
    txt_to_coco_dt(dt_int_path, dt_json_path)
    # txt_to_coco_dt2(gt_int_path, dt_json_path)
    evaluate(gt_json_path, dt_json_path)

