from __future__ import print_function
import torch.utils.data as data
from random import randint
import os
import os.path
import csv
from configure import *
from tqdm import tqdm
from PIL import Image
import os
import os.path

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class WhaleLoader(data.Dataset):
    processed_folder = 'processed'
    train_triplet_file = 'train_triplets.txt'
    test_triplet_file = 'test_triplets.txt'

    def __init__(self, config, n_triplets=50000, transform=None, loader=default_image_loader):
        # super(WhaleLoader, self).__init__()

        self.config = config
        self.transform = transform
        self.loader = loader
        self.triplets = self.make_triplet_list(n_triplets)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        img1_fn, img2_fn, img3_fn = self.triplets[index]
        img1_fn = f"{self.config.TRAIN_DIR}/{img1_fn}"
        img2_fn = f"{self.config.TRAIN_DIR}/{img2_fn}"
        img3_fn = f"{self.config.TRAIN_DIR}/{img3_fn}"

        img1 = self.loader(img1_fn)
        img2 = self.loader(img2_fn)
        img3 = self.loader(img3_fn)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1.unsqueeze(0), img2.unsqueeze(0), img3.unsqueeze(0)

    def make_triplet_list(self, ntriplets):

        # Training with labels which is not new_whale only.
        new_whale_idx = self.config.CLS_TO_IDX["new_whale"]

        print("3===D~ Generating {} triplets".format(ntriplets))

        triplets = []

        for n in tqdm(range(ntriplets)):
            anchor_idx = randint(0, self.config.N_CLASS - 1)

            # Random an anchor
            while anchor_idx == new_whale_idx:
                anchor_idx = randint(0, self.config.N_CLASS - 1)

            anchor_cls = self.config.CLASSES[anchor_idx]
            anchor_idx = randint(0, len(self.config.CLS_TO_INDICATES[anchor_cls]) - 1)
            anchor = self.config.CLS_TO_INDICATES[anchor_cls][anchor_idx]

            # Random an positive
            positive_idx = randint(0, len(self.config.CLS_TO_INDICATES[anchor_cls]) - 1)
            positive = self.config.CLS_TO_INDICATES[anchor_cls][positive_idx]

            # Random a negative
            negative_idx = randint(0, self.config.N_CLASS - 1)
            while negative_idx == anchor_idx or negative_idx == new_whale_idx:
                negative_idx = randint(0, self.config.N_CLASS - 1)

            negative_cls = self.config.CLASSES[negative_idx]
            negative_idx = randint(0, len(self.config.CLS_TO_INDICATES[negative_cls]) - 1)
            negative = self.config.CLS_TO_INDICATES[negative_cls][negative_idx]

            triplets.append((anchor, negative, positive))

        return triplets

def main():
    config = Config()
    loader = WhaleLoader(".", config, 100)


if __name__ == "__main__":
    main()
