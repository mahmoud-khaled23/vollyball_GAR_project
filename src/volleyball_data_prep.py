
import os
import cv2
import numpy as np

import pickle

import torch
import torchvision
from PIL import Image


def show_videos(path=''):
    path = path+"./volleyball_dataset/videos_g1/0"

    clip1 = "/3596"
    vid = "/3576"
    img = cv2.imread(path+clip1+vid+".jpg")
    cv2.imshow("Frame 1", img)
    cv2.waitKey()
    cv2.destroyAllWindows()



def prep_categories():
    group_categories = {
        'l-pass': 0,
        'r-pass': 1,
        'l-spike': 2,
        'r_spike': 3,
        'l_set': 4,
        'r_set': 5,
        'l_winpoint': 6,
        'r_winpoint': 7
    }

    person_categories = {
        'standing': 0,
        'setting': 1,
        'waiting': 2,
        'moving': 3,
        'falling': 4,
        'spiking': 5,
        'jumping': 6,
        'digging': 7,
        'blocking': 8
    }

    train_dirs = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31", "32", "36", "38", "39", "40",
                 "41", "42", "48", "50", "52", "53", "54"]
    val_dirs = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]

    return group_categories, person_categories


def load_video_annot(video_annot):
    with open(video_annot, 'r') as file:
        clip_category = {}

        cats = []
        for line in file:
            items = line.strip().split(' ')[:2]
            cats.append(items[1])
            clip_dir = items[0].replace('.jpg', '')
            clip_category[clip_dir] = items[1]

        return clip_category, cats


def load_videos_annots(root_annot_path, annot_encode):
    annotations = "annotations.txt"

    videos_dir = os.listdir(root_annot_path)
    videos_dir.sort()

    clips_info = []
    for video in videos_dir:
        if not os.path.isdir(os.path.join(root_annot_path, video)):
            continue

        clips_dir = os.listdir(os.path.join(root_annot_path, video))
        clips_dir.sort()

        annot_file = os.path.join(root_annot_path, video, annotations)
        clips_annot, cats = load_video_annot(annot_file)
        # print(clips_annot)

        for clip in clips_dir:
            if not os.path.isdir(os.path.join(root_annot_path, video, clip)):
                continue

            target_annot = clips_annot[clip]
            clips_annot_dct = {
                "video": video,
                "clip": clip,
                "category": annot_encode[target_annot]
            }
            clips_info.append(clips_annot_dct)

    print(clips_info)
    return clips_info

def baseline1_data_prep(root_annot_path, output_path):
    train = "train"
    val = "val"

    annot_encode = prep_categories()[0]
    data_annot = {train: load_videos_annots(os.path.join(root_annot_path, train), annot_encode),
                  val: load_videos_annots(os.path.join(root_annot_path, val), annot_encode)}

    output_file = "b1_annot.pickle"
    with open(os.path.join(output_path, output_file), "wb") as output:
        pickle.dump(data_annot, output, pickle.HIGHEST_PROTOCOL)


def annotate_clip(path):
    pass


if __name__ == '__main__':
    train_path = '/home/ma7moud-5aled/PycharmProjects/vollyball_GAR_project/volleyball_dataset/videos'
    val_path = '/home/ma7moud-5aled/PycharmProjects/vollyball_GAR_project/volleyball_dataset/videos'

    output_path = "/home/ma7moud-5aled/PycharmProjects/vollyball_GAR_project/volleyball_dataset/volleyball-baseline-annotations"
    # baseline1_data_prep(train_path, output_path)
