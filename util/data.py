import os
import cv2
import glob
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, data_root, image_size, same_id=False, same_prob=0.2, regularization_path=None):
        self.data_root = data_root
        self.image_size = image_size
        self.same_id = same_id
        self.same_prob = same_prob

        self.regularization_path = regularization_path

        self.transform = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS ),
            transforms.ToTensor()
        ])

        self.transform_recognition = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS ),
            transforms.ToTensor(),
           # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        file_list = glob.glob(os.path.join(data_root, "*", "*.*"))
        self.file_list = file_list

        self.file_list_by_dir = {}
        for file in self.file_list:
            folder = os.path.dirname(file)
            if folder not in self.file_list_by_dir.keys():
                self.file_list_by_dir[folder] = []
            self.file_list_by_dir[folder].append(file)

        if regularization_path is not None:
            self.regularization_files = glob.glob(os.path.join(self.regularization_path,"*","*.*"))
            self.len_reg = len(self.regularization_files)
        else:
            self.len_reg = 0

        self.len_main = len(self.file_list)
        self.len_total = max(self.len_main, self.len_reg)

    def __getitem__(self, index):
        if self.regularization_path is not None and random.random() <= 0.4:
            image_path = self.regularization_files[index % self.len_reg]
        else:
            image_path = self.file_list[index % self.len_main]

        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_prob:
            if self.regularization_path is not None and random.random() <= 0.4:
                image_path = random.choice(self.regularization_files)
            else:
                image_path = random.choice(self.file_list)

            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same = 0
        else:
            if self.same_id:
                image_path = random.choice(self.file_list_by_dir[os.path.dirname(image_path)])
                Xt = cv2.imread(image_path)[:, :, ::-1]
                Xt = Image.fromarray(Xt)
            else:
                Xt = Xs.copy()
            same = 1

        flip_source = False
        if random.random() < 0.2: # and same == 0:
            Xs = Xs.transpose(Image.FLIP_LEFT_RIGHT)
            flip_source = True
        flip_target = False
        if random.random() < 0.2: # and same == 0:
            Xt = Xt.transpose(Image.FLIP_LEFT_RIGHT)
            flip_target = True

        source_id = self.transform_recognition(Xs)
        source_image = self.transform(Xs)
        target_image = self.transform(Xt)

        return target_image, source_image, source_id, same

    def __len__(self):
        return self.len_total