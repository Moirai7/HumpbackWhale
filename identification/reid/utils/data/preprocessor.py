from __future__ import absolute_import
import os.path as osp
import pandas as pd
import os
from PIL import Image

class HW_Dataset(object):
    def __init__(self, filepath, csv_path=None, transform=None):
        super(HW_Dataset, self).__init__()
        self.file_path = filepath
        if csv_path is not None:
            self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_list = [x for x in os.listdir(self.file_path)]

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, idx):
        img_path = os.path.join(self.file_path, self.df.Image[idx])
        label = self.df.Id[idx]
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid