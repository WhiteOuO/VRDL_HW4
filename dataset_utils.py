import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

class PromptTrainDataset(Dataset):
    def __init__(self, args, transform=None):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.snow_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        self.transform = transform
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'desnow': 4}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'desnow' in self.de_type:
            self._init_snow_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        ref_file = self.args.data_file_dir + "noisy/denoise_airnet.txt"
        temp_ids = []
        temp_ids += [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x, "de_type": 0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x, "de_type": 1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x, "de_type": 2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_rs_ids(self):
        degraded_rain_files = [f for f in os.listdir(self.args.train_degraded_dir) if f.startswith('rain-')]
        temp_ids = [self.args.train_degraded_dir + f for f in degraded_rain_files]
        clean_rain_ids = [self.args.train_clean_dir + f.replace('rain-', 'rain_clean-') for f in degraded_rain_files]
        self.rs_ids = [{"degraded_id": temp_ids[i], "clean_id": clean_rain_ids[i], "de_type": 3} for i in range(len(temp_ids))]

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    def _init_snow_ids(self):
        degraded_snow_files = [f for f in os.listdir(self.args.train_degraded_dir) if f.startswith('snow-')]
        temp_ids = [self.args.train_degraded_dir + f for f in degraded_snow_files]
        clean_snow_ids = [self.args.train_clean_dir + f.replace('snow-', 'snow_clean-') for f in degraded_snow_files]
        self.snow_ids = [{"degraded_id": temp_ids[i], "clean_id": clean_snow_ids[i], "de_type": 4} for i in range(len(temp_ids))]

        self.snow_counter = 0
        self.num_snow = len(self.snow_ids)
        print("Total Snow Ids : {}".format(self.num_snow))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        return rainy_name.replace('rain-', 'rain_clean-')

    def _get_snow_clean_name(self, snow_name):
        return snow_name.replace('snow-', 'snow_clean-')

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids
        if "desnow" in self.de_type:
            self.sample_ids += self.snow_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id == 3:
            degrad_img = crop_img(np.array(Image.open(sample["degraded_id"]).convert('RGB')), base=16)
            clean_name = sample["clean_id"].split("/")[-1].split('.')[0]
            clean_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
        elif de_id == 4:
            degrad_img = crop_img(np.array(Image.open(sample["degraded_id"]).convert('RGB')), base=16)
            clean_name = sample["clean_id"].split("/")[-1].split('.')[0]
            clean_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)

        degrad_patch, clean_patch = self._crop_patch(degrad_img, clean_img)

        if self.transform:
            augmented = self.transform(image=degrad_patch, clean_patch=clean_patch)
            degrad_patch = augmented['image']
            clean_patch = augmented['clean_patch']

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)
    
class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_dir)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img