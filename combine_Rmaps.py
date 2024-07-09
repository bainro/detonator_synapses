import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


for condition in ["all", "without", "only"]:
    plt.figure()
    RR = cv2.imread(f'{condition}_RR_Rmap.png')[:, :, ::-1]
    RG = cv2.imread(f'{condition}_RG_Rmap.png')[:, :, ::-1]
    RB = cv2.imread(f'{condition}_RB_Rmap.png')[:, :, ::-1]
    GG = cv2.imread(f'{condition}_GG_Rmap.png')[:, :, ::-1]
    GB = cv2.imread(f'{condition}_GB_Rmap.png')[:, :, ::-1]
    BB = cv2.imread(f'{condition}_BB_Rmap.png')[:, :, ::-1]
    # Change from BGR to RGB:
    # for img in [RR, RG, RB, GG, GB, BB]:
    #     img = img[:, :, ::-1]
    WS = np.ones_like(BB) * 255 # blank White Square
    row_1 = np.hstack([RR, RG, RB])
    row_2 = np.hstack([WS, GG, GB])
    row_3 = np.hstack([WS, WS, BB])
    composite_img = np.vstack([row_1, row_2, row_3])
    plt.imshow(composite_img)
    plt.title(f'{condition}')
    # plt.show()
    img_path = os.path.join(os.getcwd(), f'{condition}_combined_Rmap.png')
    plt.savefig(img_path, dpi=300)