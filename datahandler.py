from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import cv2
import torch
import json
from torchvision import transforms, utils


class SegDataset(Dataset):
    """Segmentation Dataset"""

    def __init__(self, root_dir, imageFolder, maskFolder, transform=None, seed=None, fraction=None, subset=None, imagecolormode='rgb', maskcolormode='rgb'):
        """
        Args:
            root_dir (string): Directory with all the images and should have the following structure.
            root
            --Images
            -----Img 1
            -----Img N
            --Mask
            -----Mask 1
            -----Mask N
            imageFolder (string) = 'Images' : Name of the folder which contains the Images.
            maskFolder (string)  = 'Masks : Name of the folder which contains the Masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed: Specify a seed for the train and test split
            fraction: A float value from 0 to 1 which specifies the validation split fraction
            subset: 'Train' or 'Test' to select the appropriate set.
            imagecolormode: 'rgb' or 'grayscale'
            maskcolormode: 'rgb' or 'grayscale'
        """
        self.color_dict = {'rgb': 1, 'grayscale': 0}
        assert(imagecolormode in ['rgb', 'grayscale'])
        assert(maskcolormode in ['rgb', 'grayscale'])

        self.imagecolorflag = self.color_dict[imagecolormode]
        # self.maskcolorflag = self.color_dict[maskcolormode]
        self.root_dir = root_dir
        self.transform = transform
        self.mask_definitions = {}
        self.number_of_class = self.load_mask_definition()

        if not fraction:
            self.image_names = sorted(
                glob.glob(os.path.join(self.root_dir, imageFolder, '*')))
            self.mask_names = sorted(
                glob.glob(os.path.join(self.root_dir, maskFolder, '*')))
        else:
            assert(subset in ['Train', 'Test'])
            self.fraction = fraction
            self.image_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, imageFolder, '*'))))
            # self.mask_list = np.array(
            #     sorted(glob.glob(os.path.join(self.root_dir, maskFolder, '*'))))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                # self.mask_list = self.mask_list[indices]
            if subset == 'Train':
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list)*self.fraction))]
                # self.mask_names = self.mask_list[:int(
                #     np.ceil(len(self.mask_list)*(1-self.fraction)))]
            else:
                self.image_names = self.image_list[int(
                    np.ceil(len(self.image_list)*(1-self.fraction))):]
                # self.mask_names = self.mask_list[int(
                #     np.ceil(len(self.mask_list)*(1-self.fraction))):]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        if self.imagecolorflag:
#            image = cv2.imread(
#                img_name, self.imagecolorflag).transpose(2, 0, 1)
            image = cv2.imread(
                img_name, self.imagecolorflag)
        else:
            image = cv2.imread(img_name, self.imagecolorflag)
        image = cv2.resize(image, dsize=(480, 320), interpolation=cv2.INTER_AREA)
        image = np.transpose(image, (2, 0, 1))
        # msk_name = self.mask_names[idx]
        msk_name = img_name.replace('original', 'mask').replace('rgb', 'segmentation')

        # if self.maskcolorflag:
        #     mask = cv2.imread(msk_name, self.maskcolorflag).transpose(2, 0, 1)
        # else:
        #     png = cv2.imread(msk_name, self.maskcolorflag)
        #     mask = self.png_to_mask(png)

        try: 
            png = cv2.imread(msk_name)
            png = cv2.resize(png, dsize=(480,320), interpolation=cv2.INTER_NEAREST)
            mask = self.png_to_mask(png)
        except Exception as e:
            print(f'error mask file name is : {msk_name}')
            print(str(e))
            raise

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def png_to_mask(self, mask_image):
        mask = np.zeros(shape=(mask_image.shape[:2]), dtype=np.long)
        for index, key in enumerate(self.mask_definitions.keys()):
            value = self.mask_definitions[key]
            mask[np.all(np.abs(mask_image - value) < 2, axis=-1)] = index + 1
        return mask

    def load_mask_definition(self):
        mask_definition_json = json.load(open(os.path.join(self.root_dir, 'json/annotation_definitions.json'), 'r'))
        for spec in mask_definition_json['annotation_definitions'][1]['spec']:
            label_name = spec['label_name']
            r = round(spec['pixel_value']['r'] * 255)
            g = round(spec['pixel_value']['g'] * 255)
            b = round(spec['pixel_value']['b'] * 255)
            rgb = np.array([b, g, r], dtype=np.int64)
            self.mask_definitions[label_name] = rgb
        return len(self.mask_definitions)
# Define few transformations for the Segmentation Dataloader


class Resize(object):
    """Resize image and/or masks."""

    def __init__(self, imageresize, maskresize):
        self.imageresize = imageresize
        self.maskresize = maskresize

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)
        if len(mask.shape) == 3:
            mask = mask.transpose(1, 2, 0)
        mask = cv2.resize(mask, self.maskresize, cv2.INTER_AREA)
        image = cv2.resize(image, self.imageresize, cv2.INTER_AREA)
        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1)
        if len(mask.shape) == 3:
            mask = mask.transpose(2, 0, 1)

        return {'image': image,
                'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, maskresize=None, imageresize=None):
        image, mask = sample['image'], sample['mask']
        if len(mask.shape) == 2:
            mask = mask.reshape((1,)+mask.shape)
        if len(image.shape) == 2:
            image = image.reshape((1,)+image.shape)
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}


class Normalize(object):
    '''Normalize image'''

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': image.type(torch.FloatTensor)/255, 'mask': mask.type(torch.long)}
        # return {'image': image.type(torch.FloatTensor)/255,
        #         'mask': mask.type(torch.FloatTensor)/255}


def get_dataloader_sep_folder(data_dir, imageFolder='Image', maskFolder='Mask', batch_size=4):
    """
        Create Train and Test dataloaders from two separate Train and Test folders.
        The directory structure should be as follows.
        data_dir
        --Train
        ------Image
        ---------Image1
        ---------ImageN
        ------Mask
        ---------Mask1
        ---------MaskN
        --Train
        ------Image
        ---------Image1
        ---------ImageN
        ------Mask
        ---------Mask1
        ---------MaskN
    """
    data_transforms = {
        'Train': transforms.Compose([ToTensor(), Normalize()]),
        'Test': transforms.Compose([ToTensor(), Normalize()]),
    }

    image_datasets = {x: SegDataset(root_dir=os.path.join(data_dir, x),
                                    transform=data_transforms[x], maskFolder=maskFolder, imageFolder=imageFolder)
                      for x in ['Train', 'Test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=20)
                   for x in ['Train', 'Test']}
    return dataloaders


def get_dataloader_single_folder(data_dir, imageFolder='Images', maskFolder='Masks', fraction=0.2, batch_size=4, num_worker=0):
    """
        Create training and testing dataloaders from a single folder.
    """
    data_transforms = {
        'Train': transforms.Compose([ToTensor(), Normalize()]),
        'Test': transforms.Compose([ToTensor(), Normalize()]),
    }

    image_datasets = {x: SegDataset(data_dir, imageFolder=imageFolder, maskFolder=maskFolder, seed=100, fraction=fraction, subset=x, transform=data_transforms[x])
                      for x in ['Train', 'Test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=num_worker)
                   for x in ['Train', 'Test']}
    return dataloaders
