import sys
sys.path.append("..")
import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
from PIL import Image
import numpy as np
import albumentations as A
import cv2
import os
from MangaOCR.data.utils import get_processor

class Manga109(Dataset):
    def __init__(self, img_source ,img_text_file, processor, augment = False, max_length = 300):
        self.read_csv(img_text_file)
        self.source = img_source
        self.processor = processor
        self.augment = augment
        self.transform_medium, self.transform_heavy = self.get_transforms()
        self.max_length = max_length

    def __len__(self) -> int :
        return len(self.img_path)
    
    def __getitem__(self, index):
        img = Image.open(self.source + os.sep + self.img_path[index]).convert('RGB')
        text = self.processor.tokenizer(self.text_collection[index],
                                        padding = "max_length",
                                        max_length = self.max_length,
                                        truncation = True).input_ids
        if self.augment:
            medium_p = 0.8
            heavy_p = 0.02
            transform_variant = np.random.choice(['none', 'medium', 'heavy'],
                                                 p=[1 - medium_p - heavy_p, medium_p, heavy_p])
            transform = {
                'none': None,
                'medium': self.transform_medium,
                'heavy': self.transform_heavy,
            }[transform_variant]
        else:
            transform = None

        img = transform(image=img)['image']
        img = self.processor(img, return_tensors="pt").pixel_values
        text = np.array(text)
        # important: make sure that PAD tokens are ignored by the loss function
        text[text == self.processor.tokenizer.pad_token_id] = -100

        return img.squeeze(), torch.tensor(text)

    def read_csv(self, csv_file):
        colnames=['text','path'] 
        df = pd.read_csv(csv_file, names=colnames, header=None)
        self.img_path = df["path"]
        self.text_collection = df["text"]
    
    @staticmethod
    def get_transforms():
        t_medium = A.Compose([
            A.Rotate(5, border_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.Perspective((0.01, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.InvertImg(p=0.05),

            A.OneOf([
                A.Downscale(0.25, 0.5, interpolation=cv2.INTER_LINEAR),
                A.Downscale(0.25, 0.5, interpolation=cv2.INTER_NEAREST),
            ], p=0.1),
            A.Blur(p=0.2),
            A.Sharpen(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise((50, 200), p=0.3),
            A.ImageCompression(0, 30, p=0.1),
            A.ToGray(always_apply=True),
        ])

        t_heavy = A.Compose([
            A.Rotate(10, border_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.Perspective((0.01, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.InvertImg(p=0.05),

            A.OneOf([
                A.Downscale(0.1, 0.2, interpolation=cv2.INTER_LINEAR),
                A.Downscale(0.1, 0.2, interpolation=cv2.INTER_NEAREST),
            ], p=0.1),
            A.Blur((4, 9), p=0.5),
            A.Sharpen(p=0.5),
            A.RandomBrightnessContrast(0.8, 0.8, p=1),
            A.GaussNoise((1000, 10000), p=0.3),
            A.ImageCompression(0, 10, p=0.5),
            A.ToGray(always_apply=True),
        ])

        return t_medium, t_heavy
def train_val_split(dataset, train_size = 0.8, test_size = 0.1, val_size = 0.1):

    train_size = int(train_size * dataset.__len__())
    val_size = int(val_size * dataset.__len__())
    # test_size = int(0.1 * dataset.__len__)
    test_size = dataset.__len__() - train_size - val_size  # Ensure all instances are included
    split_sizes = [train_size, val_size, test_size]
    train_dataset, val_dataset, test_dataset = random_split(dataset, split_sizes)
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    encoder_name = 'facebook/deit-tiny-patch16-224'
    decoder_name = 'cl-tohoku/bert-base-japanese-char-v2'
    max_length = 300
    processor = get_processor(encoder_name, decoder_name)
    data = Manga109(img_text_file = "text_img.csv", processor = processor, \
                  augment=True, max_length=max_length)
