import argparse
import os
from shutil import copyfile

import pandas as pd
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True, help="dir path to images")
ap.add_argument("-o", "--output_dir", required=True, help="path to output directory")
ap.add_argument("-t", "--train", required=True, help="path to train.csv file ")
ap.add_argument("-v", "--valid", required=True, help="path to valid.csv file")

args = vars(ap.parse_args())

input_path = args['input_dir']
output_path = args["output_dir"]
train_csv_path = args["train"]
valid_csv_path = args["valid"]

dir_path = [os.path.join(output_path, suffix) for suffix in
            ['train/images', 'valid/images', 'train/labels', 'valid/labels']]

for path in dir_path:
    if not os.path.exists(path):
        os.makedirs(path)


def bbox_to_yolov5(image_w, image_h, box: list):
    # box : a list with 4 elements that contains xmin , ymin , xmax , ymax
    xmin, ymin, xmax, ymax = box
    b_center_x = (xmin + xmax) / 2
    b_center_y = (ymin + ymax) / 2
    b_width = (xmax - xmin)
    b_height = (ymax - ymin)

    # Normalise the co-ordinates by the dimensions of the image
    b_center_x /= image_w
    b_center_y /= image_h
    b_width /= image_w
    b_height /= image_h
    return b_center_x, b_center_y, b_width, b_height


def copy_images(df: pd.DataFrame, source: str, dst: str, is_test=0):
    data = df.drop_duplicates(subset=df.columns[0])

    for _, row in data.iterrows():

        copyfile(os.path.join(source, row[0]), os.path.join(dst, dir_path[is_test], row[0]))

        box_df = df[df[df.columns[0]] == row[0]]
        box_df.drop_duplicates(inplace=True)
        box_list = []
        with Image.open(os.path.join(source, row[0])) as img:
            im_width, im_height = img.size
            for idx, label in box_df.iterrows():
                center_x, center_y, width, height = bbox_to_yolov5(im_width, im_height, label[4:8])
                box_list.append("{} {} {} {} {}\n".format(0, center_x, center_y, width, height))

        im_name = row[0].split('.')[0]
        with open(os.path.join(dst, dir_path[is_test + 2], im_name + '.txt'), 'w+') as file:
            file.writelines(box_list)
    print('\npreparing proper dataset format for YOLOV5 has been finished')


train = pd.read_csv(train_csv_path)
test = pd.read_csv(valid_csv_path)

copy_images(train, input_path, output_path, is_test=0)
copy_images(test, input_path, output_path, is_test=1)
