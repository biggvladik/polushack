import os, re, json

import numpy as np
import pandas as pd

from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image
from tqdm import tqdm

import torch


def checkAndSaveTestCocoJson(model, submission_path, test_dir_path):
    coco = Coco()
    coco.add_category(CocoCategory(id=0, name='stone0'))
    coco.add_category(CocoCategory(id=1, name='stone1'))

    counter = 0
    i = 0
    for file_name in tqdm(sorted(os.listdir(test_dir_path))):
        image_id = int(re.findall(r'\d+', file_name)[0])
        image = Image.open(test_dir_path+file_name)
        coco_image = CocoImage(file_name=file_name, height=1080, width=1920, id=image_id)
        
        results = model(image, size=640)
        boxes = results.xyxy[0]
        for box in boxes:
            x_min = box[0].item()
            y_min = box[1].item()
            width = box[2].item() - x_min
            height = box[3].item() - y_min
            coco_image.add_annotation(
                CocoAnnotation(
                bbox=[x_min, y_min, width, height],
                category_id=1,
                category_name='stone1',
                image_id=image_id
                )
            )
        coco.add_image(coco_image)
        counter += 1

    save_json(data=coco.json, save_path=submission_path)


def closest_node(node, nodes):
    # Получение индекса ближайшей точки
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node) ** 2, axis = 1)
    return np.argmin(dist_2)

def get_all(true_bboxs: np.ndarray, pred_bboxs: np.ndarray):
    # Базовый расчет FP, FN
    # Получение ближайших центров квадратов для всех пар
    true = np.array([true_bboxs[:,0]+true_bboxs[:,2]/2, true_bboxs[:,1]+true_bboxs[:,3]/2]).T
    pred = np.array([pred_bboxs[:,0]+pred_bboxs[:,2]/2, pred_bboxs[:,1]+pred_bboxs[:,3]/2]).T
    
    true_cnt, pred_cnt = true.shape[0], pred.shape[0]
    if true_cnt < pred_cnt:
        FP = pred_cnt-true_cnt
        FN = 0
    elif true_cnt > pred_cnt:
        FN = true_cnt-pred_cnt
        FP = 0
    else:
        FP = FN = 0
    closet_inds = []
    if FP>=FN:
        for i in true:
            closet_inds.append(closest_node(i, pred))
    else:
        for i in pred:
            closet_inds.append(closest_node(i, true))
    return closet_inds, FP, FN

def bb_intersection_over_union(boxA_:np.ndarray, boxB_:np.ndarray):
    # Расчет IoU
    boxA, boxB = boxA_.copy(), boxB_.copy()
    # Корректировка формата (x,y,w,h) -> (x1,y1,x2,y2)
    boxA[2], boxB[2], boxA[3], boxB[3] = boxA[0]+boxA[2], boxB[0]+boxB[2], boxA[1]+boxA[3], boxB[1]+boxB[3]
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# Юзать этот метод, он принимает два файла, который являются байтовыми словарями
def run(bytes_str_true, bytes_str_sub):
    annot = json.loads(bytes_str_true.decode("utf-8").replace("'",'"'))

    df_images_true = pd.DataFrame(annot['images'])
    df_annot_true = pd.DataFrame(annot['annotations'])

    annot = json.loads(bytes_str_sub.decode("utf-8").replace("'",'"'))

    df_images_sub = pd.DataFrame(annot['images'])
    df_annot_sub = pd.DataFrame(annot['annotations'])

    # обрабатываем ситуацию, когда пара file_name -- id
    # не соответствует в оригинальных и прогнозных файлах
    # df_images_true = df_images_true.set_index('file_name').sort_index()
    # df_images_sub = df_images_sub.set_index('file_name').sort_index()

    # df_images_sub.loc[:,'id']=df_images_true['id'].values


    iou=[]
    FP = FN =  0
    # Внешний цикл, проходит по всем кадрам
    for img_id in df_images_true['id'].values:
        true = np.array(list(df_annot_true[df_annot_true['image_id']==img_id]['bbox'].values))
        pred = np.array(list(df_annot_sub[df_annot_sub['image_id']==img_id]['bbox'].values))

        coord_inds, FP, FN = get_all(true, pred)
        # Внутренний цикл, считает IoU для пар
        true_ = []
        pred_ = []
        for i,j in enumerate(coord_inds):
            if FN>+FP:
                iou.append(bb_intersection_over_union(true[j], pred[i]))
            else:
                iou.append(bb_intersection_over_union(pred[j], true[i]))


    iou = np.array(iou)
    FP += np.sum(iou<=0.5)
    TP = np.sum(iou>0.5)
    mIoU=np.mean(iou)
    beta = 2
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    fb = (((1+beta**2)*precision*recall)/((beta**2)*precision+recall))*mIoU
    print("FP:", FP, "FN:", FN, "mIoU:", mIoU, "Fbeta:", (((1+beta**2)*precision*recall)/((beta**2)*precision+recall)))
    print("Final score: ",fb) # ИТоговый результат метрики

if __name__ == "__main__":
    device = "cuda:0"
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/yolov5s10epochstrainonly.pt', force_reload=True, device=device)
    #model.conf = 0.7
    #model.iou = 0.8
    checkAndSaveTestCocoJson(model, "valid.json", "datasets/particles/images/test/")

    # Score
    f = open('original_dataset/annot_local/test_annotation.json')
    annot = json.load(f)
    bytes_str_true = json.dumps(annot, indent=2).encode('utf-8') 

    # загружаем прогнозные аннотации json и забираем необходимые части
    f = open('valid.json')
    annot = json.load(f)
    bytes_str_sub = json.dumps(annot, indent=2).encode('utf-8')
    run(bytes_str_true, bytes_str_sub)


    checkAndSaveTestCocoJson(model, "submission_private_reload.json", "datasets/particles/images/private/")