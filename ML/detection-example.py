import time
import cv2
import numpy as np
import math

import torch

import sys
sys.path.append("yolov5/")

from yolov5.models.common import Detections
from yolov5.utils.plots import Annotator, colors, save_one_box

from PIL import Image
from pathlib import Path


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/yolov5s10epochstrainonly.pt', force_reload=False, device=device)
#model.conf = 0.5
#model.iou = 0.95
cap = cv2.VideoCapture("video/example.mp4")

FPS = 30
TIME_PER_FRAME = 1000 / FPS # In ms

GREEN_AREA_MIN_Y = 580
GREEN_AREA_MAX_Y = 850
GREEN_AREA_TOP_MIN_X = 460
GREEN_AREA_TOP_MAX_X = 1130
GREEN_AREA_BOTTOM_MIN_X = 225
GREEN_AREA_BOTTOM_MAX_X = 1325

GREEN_AREA_TOP_WIDTH = GREEN_AREA_TOP_MAX_X - GREEN_AREA_TOP_MIN_X
GREEN_AREA_BOTTOM_WIDTH = GREEN_AREA_BOTTOM_MAX_X - GREEN_AREA_BOTTOM_MIN_X
GREEN_AREA_HEIGHT = GREEN_AREA_MAX_Y - GREEN_AREA_MIN_Y

WIDTH_IN_MM = 1600
NEGABARIT_IN_MM = 600

LEFT_SIDE_LENGTH = math.sqrt((GREEN_AREA_TOP_MIN_X - GREEN_AREA_BOTTOM_MIN_X) ** 2 + (GREEN_AREA_MAX_Y - GREEN_AREA_MIN_Y) ** 2)
RIGHT_SIDE_LENGTH = math.sqrt((GREEN_AREA_TOP_MAX_X - GREEN_AREA_BOTTOM_MAX_X) ** 2 + (GREEN_AREA_MAX_Y - GREEN_AREA_MIN_Y) ** 2)
SIDE_TO_HEIGHT_RATIO = (LEFT_SIDE_LENGTH + RIGHT_SIDE_LENGTH ) / (2 * GREEN_AREA_HEIGHT)

GREEN_AREA_AREA = (GREEN_AREA_TOP_WIDTH + GREEN_AREA_BOTTOM_WIDTH) * GREEN_AREA_HEIGHT / 2

HEIGHT_IN_MM = 1600

def _run_override(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
    # Это оверрайд метода из йолов5

    ## Per frame metrics 
    negabarit_found = False
    empty_line = False
    predicted_sizes = []
    predicted_areas = []
    

    s, crops = '', []
    for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
        im = draw_accept_area(im)
        s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
        if pred.shape[0]:
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
            s = s.rstrip(', ')
            if show or save or render or crop:
                annotator = Annotator(im, example=str(self.names), line_width=2)

                


                if len(pred) == 0:
                    empty_line = True

                for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                    # Делаем что надо с боксами
                    # Вычисляем, попал ли бокс в зеленую зону
                    x_min, y_min, x_max, y_max = box
                    if y_min < GREEN_AREA_MAX_Y and \
                        y_max > GREEN_AREA_MIN_Y and \
                        y_min > GREEN_AREA_MIN_Y and \
                        y_max < GREEN_AREA_MAX_Y:
                        box_inside = True
                        new_cls = cls
                    else:
                        box_inside = False
                        new_cls = cls

                    # Если попал:
                    if box_inside:
                        # Вычисляем реальный размер бокса с учетом перспективы
                        # Ширина (Ширина бокса / Длину линии трапеции, на которой он находится)
                        box_width = x_max - x_min
                        box_height = y_max - y_min
                        h = y_max - GREEN_AREA_MAX_Y
                        trapezoid_line_width = GREEN_AREA_BOTTOM_WIDTH - (GREEN_AREA_BOTTOM_WIDTH - GREEN_AREA_TOP_WIDTH) * \
                            h / GREEN_AREA_HEIGHT

                        real_width = box_width / trapezoid_line_width * WIDTH_IN_MM
                        # Высота
                        real_height = (box_height / SIDE_TO_HEIGHT_RATIO) / GREEN_AREA_HEIGHT * HEIGHT_IN_MM
                        
                        real_size = max(real_width, real_height) * 0.9

                        if real_size > NEGABARIT_IN_MM:
                            # Change color
                            new_cls = 14
                            negabarit_found = True

                        predicted_sizes.append(real_size)
                        predicted_areas.append(box_height*box_width)



                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(box, label if labels else '', color=colors(new_cls))

                im = annotator.im
        else:
            s += '(no detections)'

        im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
        if render:
            self.ims[i] = np.asarray(im)

    return self.ims, negabarit_found, empty_line, predicted_sizes, predicted_areas


def draw_accept_area(image):
    pts = np.array([[GREEN_AREA_TOP_MIN_X, GREEN_AREA_MIN_Y], [GREEN_AREA_TOP_MAX_X, GREEN_AREA_MIN_Y ],
                [GREEN_AREA_BOTTOM_MAX_X, GREEN_AREA_MAX_Y], [GREEN_AREA_BOTTOM_MIN_X, GREEN_AREA_MAX_Y]],
               np.int32)
 
    pts = pts.reshape((-1, 1, 2))

    isClosed = True
    
    # Color in BGR
    color = (0, 255, 0)
    
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.polylines() method
    # Draw a Blue polygon with
    # thickness of 1 px
    return cv2.polylines(image, [pts],
                        isClosed, color, thickness)

def get_histogram(particle_sizes):
    histogram = {i: 0 for i in range(1, 8)}

    for size_ in particle_sizes:
        if size_ <= 40:
            histogram[7] += 1
        elif size_ <= 70:
            histogram[6] += 1
        elif size_ <= 80:
            histogram[5] += 1
        elif size_ <= 100:
            histogram[4] += 1
        elif size_ <= 150:
            histogram[3] += 1
        elif size_ <= 250:
            histogram[2] += 1
        else:
            histogram[1] += 1

    for i in range(1, 8):
        histogram[i] /= len(particle_sizes) 

    return histogram

# Read until video is completed
while(cap.isOpened()):
    frame_beggining_time = time.time() * 1000
    frame_counter = 0
    # Capture frame-by-frame
    ret, frame = cap.read()


    ## Metrics
    particle_sizes_arr = [] # Среднее + макс каждую секунду
    large_particle_percentage = [] # Процент площади крупных частиц каждую секунду
    negabarit_this_second = False # Был ли в эту секунду негабарит



    if ret == True:
        results = model(frame, size=1280)
        results._run = _run_override.__get__(results, Detections)
        ims, negabarit_found, empty_line, predicted_sizes, predicted_areas = results._run(labels=False, render=True)
        particle_sizes_arr += predicted_sizes
        large_particle_percentage.append((sum(predicted_areas) * 0.75) / (GREEN_AREA_AREA * 0.85))

        if negabarit_found:
            negabarit_this_second = True

        cv2.imshow('Frame', ims[0])

        processing_time = time.time() * 1000 - frame_beggining_time
        print(f"Processing took: {int(processing_time)} ms")

        # How much to wait until showing next frame
        after_frame_wait = TIME_PER_FRAME - processing_time

        if after_frame_wait <= 1:
            after_frame_wait = 1

        frame_counter += 1
        # Если началась следующая секунда
        # Пересчет ежесекундных метрик
        if frame_counter % FPS == 0:
            average_predicted_size = sum(particle_sizes_arr) / len(particle_sizes_arr)
            average_large_particle_percentage = sum(large_particle_percentage) / len(large_particle_percentage)
            max_predicted_size = max(particle_sizes_arr)
            # Гистограмма
            histogram = get_histogram(particle_sizes_arr)

            # Обнуляем все
            particle_sizes_arr = []
            large_particle_percentage = []
            negabarit_this_second = False


        # ВСЕ ДОСТУПНЫЕ ДАННЫЕ!!
        # ims[0] - картинка, каждый фрейм
        # negabarit_found, empty_line - булевые флаги, каждый фрейм
        # average_predicted_size - средний размер частиц в мм, каждую секунду
        # max_predicted_size - максимальный размер частиц, каждую секунду
        # average_large_particle_percentage - средний процент больших частиц по отношению к малым, каждую секунду
        # histogram - dict с процентами классов, каждую секунду
        # negabarit_this_second - булевый флаг был ли негабарит в эту секунду

        # Press Q on keyboard to exit
        if cv2.waitKey(int(after_frame_wait)) & 0xFF == ord('q'):
            break
    
    # Break the loop
    else:
        break
 
# When everything done, release
# the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
