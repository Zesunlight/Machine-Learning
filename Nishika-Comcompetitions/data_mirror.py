# -*- coding: UTF-8 -*-
"""
=================================================
@Project: how_to_do
@File   : data_mirror
@IDE    : PyCharm
@Author : Zhao Yongze
@Date   : 20/6/16
@Des    : in vain
=================================================="""

from PIL import Image
import os
import glob
import cv2
import csv
from tqdm import tqdm


def image_flip():
    data_path = r''
    for image_path in tqdm(glob.glob(os.path.join(data_path, '*.jpg'))):
        # print(image_path)
        # img = Image.open(image_path).convert("RGB")
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # img.save('test.jpg', "JPEG")

        img2 = cv2.imread(image_path)
        img2 = cv2.flip(img2, 1)
        save_path = os.path.join(r'', os.path.basename(image_path)[:-4] + '_f.jpg')
        cv2.imwrite(save_path, img2)


def make_label_csv():
    label_path = r''
    with open(label_path, 'r') as file:
        with open(r'', 'w', newline='') as label:
            f_csv = csv.writer(label)
            f_csv.writerow(['image', 'gender_status'])

            lines = csv.reader(file, delimiter=',')
            next(lines)  # ignore the first line
            for line in lines:
                f_csv.writerow(line)
                f_csv.writerow([line[0][:-4] + '_f.jpg', line[1]])
    return


if __name__ == '__main__':
    # image_flip()
    # make_label_csv()
    print(os.path.basename(r''))
