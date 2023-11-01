import copy
import math
import os
import random

import cv2
import numpy
import torch
from PIL import Image
from PIL import ImageFilter


def process(data_dir, folder, image_name, label_name, target_size):
    image_path = os.path.join(data_dir, folder, image_name)
    label_path = os.path.join(data_dir, folder, label_name)

    with open(label_path, 'r') as f:
        if folder.split('/')[0] == 'dv2':
            annotation = f.readlines()[3:-1]
        elif folder.split('/')[0] == 'dibox2' or folder.split('/')[0] == 'nir_face2':
            annotation = f.readlines()
        elif folder.split('/')[0] == 'prevent':
            annotation = f.readlines()[3:-3]
        elif folder.split('/')[0] == 'zerone2':
            annotation = f.readlines()[3:-1]
        else:
            print("Error check label")

        annotation = [x.strip().split() for x in annotation]
        annotation = [[int(float(x[0])), int(float(x[1]))] for x in annotation]

        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        anno_x = [x[0] for x in annotation]
        anno_y = [x[1] for x in annotation]
        x_min = min(anno_x)
        y_min = min(anno_y)
        x_max = max(anno_x)
        y_max = max(anno_y)
        box_w = x_max - x_min
        box_h = y_max - y_min
        scale = 1.1
        x_min -= int((scale - 1) / 2 * box_w)
        y_min -= int((scale - 1) / 2 * box_h)
        box_w *= scale
        box_h *= scale
        box_w = int(box_w)
        box_h = int(box_h)
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        box_w = min(box_w, image_width - x_min - 1)
        box_h = min(box_h, image_height - y_min - 1)
        annotation = [[(x - x_min) / box_w, (y - y_min) / box_h] for x, y in annotation]

        x_max = x_min + box_w
        y_max = y_min + box_h
        image_crop = image[y_min:y_max, x_min:x_max, :]
        image_crop = cv2.resize(image_crop, (target_size, target_size))
        return image_crop, annotation


def convert(data_dir, target_size=256):
    if not os.path.exists(os.path.join(data_dir, 'images', 'train')):
        os.makedirs(os.path.join(data_dir, 'images', 'train'))
    if not os.path.exists(os.path.join(data_dir, 'images', 'test')):
        os.makedirs(os.path.join(data_dir, 'images', 'test'))

    folders = ['dv2/train', 'dibox2/train', 'nir_face2/train', 'prevent/train', 'zerone2/train']
    annotations = {}
    for folder in folders:
        filenames = sorted(os.listdir(os.path.join(data_dir, folder)))
        label_files = [x for x in filenames if '.pts' in x]
        image_files = [x for x in filenames if '.pts' not in x]
        assert len(image_files) == len(label_files)
        for image_name, label_name in zip(image_files, label_files):
            image_crop_name = folder.replace('/', '_') + '_' + image_name
            image_crop_name = os.path.join(data_dir, 'images', 'train', image_crop_name)

            image_crop, annotation = process(data_dir, folder, image_name, label_name, target_size)
            cv2.imwrite(image_crop_name, image_crop)
            annotations[image_crop_name] = annotation
    with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
        for image_crop_name, annotation in annotations.items():
            f.write(image_crop_name + ' ')
            for x, y in annotation:
                f.write(str(x) + ' ' + str(y) + ' ')
            f.write('\n')

    annotations = {}
    folders = ['dv2/test', 'dibox2/test', 'nir_face2/test', 'prevent/test', 'zerone2/test']
    for folder in folders:
        filenames = sorted(os.listdir(os.path.join(data_dir, folder)))
        label_files = [x for x in filenames if '.pts' in x]
        image_files = [x for x in filenames if '.pts' not in x]
        assert len(image_files) == len(label_files)
        for image_name, label_name in zip(image_files, label_files):
            image_crop_name = folder.replace('/', '_') + '_' + image_name
            image_crop_name = os.path.join(data_dir, 'images', 'test', image_crop_name)

            image_crop, annotation = process(data_dir, folder, image_name, label_name, target_size)
            cv2.imwrite(image_crop_name, image_crop)
            annotations[image_crop_name] = annotation
    with open(os.path.join(data_dir, 'test.txt'), 'w') as f:
        for image_crop_name, annotation in annotations.items():
            f.write(image_crop_name + ' ')
            for x, y in annotation:
                f.write(str(x) + ' ' + str(y) + ' ')
            f.write('\n')

    with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
        annotations = f.readlines()
    with open(os.path.join(data_dir, 'test_common.txt'), 'w') as f:
        for annotation in annotations:
            if 'ibug' not in annotation:
                f.write(annotation)
    with open(os.path.join(data_dir, 'test_challenge.txt'), 'w') as f:
        for annotation in annotations:
            if 'ibug' in annotation:
                f.write(annotation)

    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        annotations = f.readlines()
    annotations = [x.strip().split()[1:] for x in annotations]
    annotations = [[float(x) for x in anno] for anno in annotations]
    annotations = numpy.array(annotations)
    mean_face = [str(x) for x in numpy.mean(annotations, axis=0).tolist()]

    with open(os.path.join(data_dir, 'indices.txt'), 'w') as f:
        f.write(' '.join(mean_face))


data_dir = '/mnt/data/Projects/Datasets/IR/'
convert(data_dir)


# file_path = '/mnt/data/Projects/Datasets/IR/test.txt'


# def analyze_points(file_path):
#     with open(file_path, 'r') as file:
#         i = 0
#         for line in file:
#             # Splitting the line into image path and points
#             parts = line.strip().split(' ')
#             if len(parts) != 137:
#                 i += 1
#                 print(parts[0])
#         print(i)
#
#
# analyze_points(file_path)


