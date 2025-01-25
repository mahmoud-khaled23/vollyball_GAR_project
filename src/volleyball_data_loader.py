import os
import pickle

import cv2
import numpy as np

from src.volleyball_data_prep import prep_categories
import torchvision.transforms as transforms

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def load_video_annot(video_annot):
    with open(video_annot, 'r') as file:
        clip_category = {}

        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category[clip_dir] = items[1]

        return clip_category


class VolleyBallDataSet(Dataset):

    def __init__(self, root_videos_path, annot_dct, preprocess=None):
        # super().__init__(self)
        self.root_videos_path = root_videos_path
        self.annot_dct = annot_dct
        self.preprocess = preprocess


    def __getitem__(self, idx):
        video = self.annot_dct[idx]["video"]
        clip = self.annot_dct[idx]["clip"]
        category =  self.annot_dct[idx]["category"]

        image_path = os.path.join(self.root_videos_path, video, clip, f'{clip}.jpg')
        image = Image.open(image_path).convert('RGB')

        if self.preprocess:
            image = self.preprocess(image)

        category = torch.tensor(category)

        return image, category

    def __len__(self):
        return len(self.annot_dct)


if __name__ == '__main__':
    # print(os.listdir('.'))
    root_dataset = '/home/ma7moud-5aled/PycharmProjects/vollyball_GAR_project/volleyball_dataset/'
    train_path = root_dataset+'videos/train'
    val_path = root_dataset+'videos/val'

    annot_pkl_path = root_dataset+"volleyball-baseline-annotations/b1_annot.pickle"

    grop = prep_categories()[0]
    print(grop['l-pass'])

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(annot_pkl_path, 'rb') as file:
        data_annot = pickle.load(file)

    train_annot = data_annot["train"]
    val_annot = data_annot["val"]

    train_data = VolleyBallDataSet(train_path, train_annot, preprocess=preprocess)
    val_data = VolleyBallDataSet(val_path, val_annot, preprocess=preprocess)

    img, cat = train_data[0]
    print(f'image shape: {np.array(img).shape}, category: {cat}')

    cv2.imshow("Image", np.array(img.permute(1, 2, 0)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print(f'image : {np.array(img)}, category: {cat}')
    print(f'image shape: {np.array(img).shape}, category: {cat}')
