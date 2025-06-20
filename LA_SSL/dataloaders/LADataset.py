import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from torchvision.transforms import Compose
import logging


class LAHeart(Dataset):
    """ LA Dataset """

    def __init__(self, data_dir, list_dir, split, reverse=False, aug_times=1, logging=None):
        print("✅ LAHeart class with logging support loaded.")
        self.data_dir = data_dir + "/Training Set"
        self.list_dir = list_dir
        self.split = split
        self.reverse = reverse
        self.aug_times = aug_times
        self.logger = logging

        tr_transform = Compose([
            RandomCrop((112, 112, 80)),
            ToTensor()
        ])
        test_transform = Compose([
            CenterCrop((112, 112, 80)),
            ToTensor()
        ])

        if split in ['train_lab', 'lab']:
            data_path = os.path.join(list_dir, 'train_lab.txt')
            self.transform = tr_transform
        elif split in ['train_unlab', 'unlab']:
            data_path = os.path.join(list_dir, 'train_unlab.txt')
            self.transform = tr_transform
            print("unlab transform")
        elif split in ['train']:
            data_path = os.path.join(list_dir, 'train.txt')
            self.transform = tr_transform
        else:
            data_path = os.path.join(list_dir, 'test.txt')
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = [line.strip() for line in f.readlines()]

        self.image_list = [os.path.join(self.data_dir, item, "mri_norm2.h5") for item in self.image_list]

        if self.logger:
            self.logger.info("{} set: total {} samples".format(split, len(self.image_list)))
            self.logger.info("Sample paths: {}".format(self.image_list))

    def __len__(self):
        if self.split in ["train_lab", "train_unlab", "lab", "unlab", "train"]:
            return len(self.image_list) * self.aug_times
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        if self.reverse:
            image_path = self.image_list[len(self.image_list) - idx % len(self.image_list) - 1]
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
        samples = image, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_, label_ = tr_samples
        return {"image": image_.float(), "label": label_.long()}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomCrop(object):
    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class ToTensor(object):
    def __call__(self, sample):
        image = sample[0]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample = [image] + [*sample[1:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]


if __name__ == "__main__":
    data_dir = "/content/drive/MyDrive/0SSL/Dataset/2018_UTAH_MICCAI"
    list_dir = "/content/drive/MyDrive/0SSL/rakibiuict/LA_SSL/Datasets/la/data_split"

    print("\n--- Train Split ---")
    train_dataset = LAHeart(data_dir, list_dir, split="train_lab")
    print("Total train samples:", len(train_dataset))
    sample = train_dataset[0]
    print("Train image shape:", sample["image"].shape)
    print("Train label shape:", sample["label"].shape)

    print("\n--- Test Split ---")
    test_dataset = LAHeart(data_dir, list_dir, split="test")
    print("Total test samples:", len(test_dataset))
    sample = test_dataset[0]
    print("Test image shape:", sample["image"].shape)
    print("Test label shape:", sample["label"].shape)
