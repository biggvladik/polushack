import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from torchvision.io import read_image

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F



def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])




img = read_image("datasets/particles/images/test/frame1800.jpg")

show(img)
plt.show()

true_boxes = [[533, 458, 533+51, 458+70],
              [508, 378, 508+74, 378+48]]

pred_boxes = [[479, 443, 479+87, 443+68],
              [541, 405, 541+74, 405+56]]

drawn_boxes = draw_bounding_boxes(img, torch.Tensor(true_boxes), colors="red")
drawn_boxes = draw_bounding_boxes(drawn_boxes, torch.Tensor(pred_boxes), colors="blue")
show(drawn_boxes)
plt.show()