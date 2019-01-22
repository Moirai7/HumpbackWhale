from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re
from PIL import Image
import os

class HW_Dataset(Dataset):
    def __init__(self, filepath, csv_path, transform=None):
        self.file_path = filepath
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_list = [x for x in os.listdir(self.file_path)]

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, idx):
        self.df["Image"] = self.df["Image"].map(lambda x: "../dataset/train/" + x)
        ImageToLabelDict = dict(zip(self.df["Image"], self.df["Id"]))
        img_path = os.path.join(self.file_path, self.df.Image[idx])
        label = self.df.Id[idx]

        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label,ImageToLabelDict



