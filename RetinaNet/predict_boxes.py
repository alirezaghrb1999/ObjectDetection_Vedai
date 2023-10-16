# get test images path , make prediction on them and output predicted boxes and classes

import argparse
import csv
import os

import numpy as np
import pandas as pd

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True, help="dir path to inference images")
ap.add_argument("-t", "--threshold", default=0.5, type=float, help="threshold for filtering weak detections")
ap.add_argument("-m", "--model", required=True, help="path to trained/converted model")
ap.add_argument("-o", "--output_dir", required=True, help="path to output directory")
ap.add_argument("-l", "--labels", required=True, help="path to class csv")
ap.add_argument("-c", "--csv_input", required=True, help="path to csv test images")

args = vars(ap.parse_args())

input_path = args['input_dir']
output_path = args["output_dir"]
thresh_score = args["threshold"]

LABELS = open(args["labels"]).read().strip().split('\n')
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

model = models.load_model(args["model"], backbone_name='resnet50')

df = pd.read_csv(args["csv_input"])
img_name = df[df.columns[0]].tolist()

inference_images = [os.path.join(input_path, jpg_file) for jpg_file in img_name]

output_file = os.path.join(output_path, 'out.csv')

# loop over test images to predict them
for (i, img_path) in enumerate(inference_images):
    # load image and preprocess it according the way datas trained on model
    image = read_image_bgr(img_path)
    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis=0)

    (boxes, scores, labels) = model.predict_on_batch(image)
    boxes /= scale

    items = []

    # loop over all boxes outputs from model
    for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
        # weak detections are filtered
        if score < thresh_score:
            continue

        b = box.astype(int)
        item = [img_name[i], b[0], b[1], b[2], b[3], LABELS[label], score]
        # add detection to list
        items.append(item)

    with open(output_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(items)

print("[FINAL] Predictions completed!")
