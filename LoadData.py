from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import random

class ImageDataset(Dataset):
    def __init__(self, imageFilenames, labels, npx):
        self.imageFilenames = imageFilenames
        self.labels = labels 
        self.npx = npx

    def __len__(self):
        return len(self.imageFilenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = makeIm(self.imageFilenames[idx])
        x = randomSquareCrop(x, self.npx)
        sample = (x, self.labels[idx])
        return sample


def randomSquareCrop(im, size):
    # Im is [nc, w, h ] 
    w = im.shape[1]
    h = im.shape[2]

    xoffset = random.randint(0, w - size)
    yoffset = random.randint(0, h - size)

    return im[:, xoffset: xoffset + size, yoffset: yoffset + size]


def makeIm(filename):
    im = Image.open(filename)
    imnp = np.array(im)
    channels = [imnp[:, :, i] for i in range(3)]
    channels = np.stack(channels, axis=0)
    channels = channels.astype('float')
    channels = channels / 255.0 * 2 - 1
    
    return channels

def loadDataset(version='train', npx=128):
    # The CSV file should be in the format:
    # name_without_extension
    # flower1
    # flower2
    # flowerbig
    # etc

    # The script automatically adds an extension (.png)
    
    ref = open(f'{version}/{version}_labels.csv').readlines()[1:]

    imageFilenames = []
    labels = []
    ct = 0
    for line in ref:
        name = line.rstrip('\n')
        name = f"{version}/{name}.png"
        try:
            a = open(name)
            imageFilenames.append(name)
            labels.append(ct)
            ct += 1
        except FileNotFoundError:
            pass

    print(f"{ct} images found in dir")

    dataset = ImageDataset(imageFilenames, labels, npx)

    return dataset

def examine(image):
    print(f"Shape {image.shape}")
    print(image[:,:,:])

def test():
    dataset = loadDataset("dataset", 257)
    im, label = dataset[0]
    examine(im)


if __name__ == "__main__":
    test()
