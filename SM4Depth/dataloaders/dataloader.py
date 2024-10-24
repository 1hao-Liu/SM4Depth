import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image
import os, cv2
import h5py
import random

from utils import DistributedSamplerNoEvenlyDivisible, WeightBatchSampler, auto_fov_fitting, auto_fov_fitting_test, DistributedWeightBatchSampler


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class SM4DataLoader(object):
    def __init__(self, args, mode, val_path=None):
        if mode == 'train':
            # The training code will be released.

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode), is_for_online_eval=False, val_path=val_path)
            if args.distributed:
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=2,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False, val_path = None):
        self.args = args
        self.val_path = val_path
        if mode == 'online_eval':
            with open(val_path, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def getType(self):
        label = []
        for idx in range(len(self.filenames)):
            sample_path = self.filenames[idx]
            depth_type = int(sample_path.split()[2]) if len(sample_path.split()) > 2 else 0
            label.append(depth_type)
        return torch.tensor(label)

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        if self.mode == 'train':
            image_path = sample_path.split()[0]
            depth_path = sample_path.split()[1]
            depth_type = int(sample_path.split()[2]) if len(sample_path.split()) > 2 else 0

            if 'Hypersim' in image_path:
                image = Image.open(image_path).resize((539, 398), Image.ANTIALIAS)
                image = np.asarray(image, dtype=np.float32) / 255.0
                image = np.pad(image, ((13, 13), (13, 13), (0, 0)), 'constant', constant_values=(0))
                image[:13, 13:552] = image[13:26, 13:552][::-1, :]
                image[-13:, 13:552] = image[-26:-13, 13:552][::-1, :]
                image[:, :13] = image[:, 13:26][:, ::-1]
                image[:, -13:] = image[:, -26:-13][:, ::-1]
                depth_gt = np.asarray(h5py.File(depth_path, 'r')['dataset'], dtype=np.float32)
                depth_gt = cv2.resize(depth_gt, dsize=(539, 398), interpolation=cv2.INTER_NEAREST)
                depth_gt = np.pad(depth_gt, ((13, 13), (13, 13)), 'constant', constant_values=(0))
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'DIML/indoor' in image_path:
                image = Image.open(image_path).crop((79, 0, 1186, 755)).resize((640, 408), Image.ANTIALIAS)
                depth_gt = Image.open(depth_path).crop((79, 0, 1186, 755)).resize((640, 408), Image.NEAREST)
                image = np.asarray(image, dtype=np.float32) / 255.0
                image = np.pad(image, ((8, 8), (0, 0), (0, 0)), 'constant', constant_values=(1))
                image[:8, :] = image[8:16, :][::-1, :]
                image[-8:, :] = image[-16:-8, :][::-1, :]
                image = image[:, 44:608]
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 1000.0
                depth_gt = np.pad(depth_gt, ((8, 8), (0, 0)), 'constant', constant_values=(0))
                depth_gt = np.expand_dims(depth_gt, axis=2)
                depth_gt = depth_gt[:, 44:608]
            elif 'nuscenes' in image_path:
                image = Image.open(image_path).crop((102, 0, 1499, 899)).resize((640, 412), Image.ANTIALIAS)
                depth_gt = Image.open(depth_path).crop((102, 0, 1499, 899)).resize((640, 412), Image.NEAREST)
                random_angle = (random.random() - 0.5) * 2
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
                image = np.asarray(image, dtype=np.float32) / 255.0
                image = np.pad(image, ((12, 0), (0, 0), (0, 0)), 'constant', constant_values=(1))
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 256.0
                depth_gt = np.pad(depth_gt, ((12, 0), (0, 0)), 'constant', constant_values=(0))
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'leftImg8bit_trainextra' in image_path:
                mask_path = sample_path.split()[3]
                image = Image.open(image_path).resize((515, 256), Image.ANTIALIAS)
                depth_gt = Image.open(depth_path).resize((515, 256), Image.NEAREST)
                mask = Image.open(mask_path).resize((515, 256), Image.NEAREST)
                random_angle = (random.random() - 0.5) * 2
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
                mask = self.rotate_image(mask, random_angle, flag=Image.NEAREST)
                image = np.asarray(image, dtype=np.float32) / 255.0
                try:
                    image = np.pad(image, ((171, 0), (25, 25), (0, 0)), 'constant', constant_values=(1))
                except:
                    print(image_path)
                    print('1')
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 256.0
                mask = np.asarray(mask)
                depth_gt[mask] = 0
                depth_gt = np.pad(depth_gt, ((171, 0), (25, 25)), 'constant', constant_values=(0))
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'leftImg8bit_trainvaltest' in image_path:
                mask_path = sample_path.split()[3]
                image = Image.open(image_path).resize((515, 256), Image.ANTIALIAS)
                depth_gt = Image.open(depth_path).resize((515, 256), Image.NEAREST)
                mask = Image.open(mask_path).resize((515, 256), Image.NEAREST)
                random_angle = (random.random() - 0.5) * 2
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
                mask = self.rotate_image(mask, random_angle, flag=Image.NEAREST)
                image = np.asarray(image, dtype=np.float32) / 255.0
                try:
                    image = np.pad(image, ((171, 0), (25, 25), (0, 0)), 'constant', constant_values=(1))
                except:
                    print(image_path)
                    print('2')
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 256.0
                mask = np.asarray(mask, dtype=np.float32)
                mask_nolabel = np.logical_and(np.logical_and(mask[:,:,0]==0, mask[:,:,1]==0), mask[:,:,2]==0)
                mask_sky = np.logical_and(np.logical_and(mask[:,:,0]==70, mask[:,:,1]==130), mask[:,:,2]==180)
                mask_no = np.logical_or(mask_nolabel, mask_sky)
                depth_gt[mask_no] = 0
                depth_gt = np.pad(depth_gt, ((171, 0), (25, 25)), 'constant', constant_values=(0))
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'DIML_indoor' in image_path:
                try:
                    # image = Image.open(image_path).crop((180, 0, 1228, 791)).resize((565, 427), Image.ANTIALIAS)
                    image = Image.open(image_path).crop((110, 0, 1298, 791)).resize((640, 427), Image.ANTIALIAS)
                except:
                    print(image_path)
                    print('3')
                # depth_gt = Image.open(depth_path).crop((180, 0, 1228, 791)).resize((565, 427), Image.NEAREST)
                depth_gt = Image.open(depth_path).crop((110, 0, 1298, 791)).resize((640, 427), Image.NEAREST)
                image = np.asarray(image, dtype=np.float32) / 255.0
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 1000.0
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'DIML_outdoor' in image_path:
                mask_path = sample_path.split()[3]
                # image = Image.open(image_path).crop((271, 20, 1649, 1060)).resize((565, 427), Image.ANTIALIAS)
                # depth_gt = Image.open(depth_path).crop((271, 20, 1649, 1060)).resize((565, 427), Image.NEAREST)
                # mask_sky = Image.open(mask_path).crop((271, 20, 1649, 1060)).resize((565, 427), Image.NEAREST)
                image = Image.open(image_path).crop((180, 0, 1740, 1079)).resize((640, 443), Image.ANTIALIAS)
                depth_gt = Image.open(depth_path).crop((180, 0, 1740, 1079)).resize((640, 443), Image.NEAREST)
                mask_sky = Image.open(mask_path).crop((180, 0, 1740, 1079)).resize((640, 443), Image.NEAREST)
                image = np.asarray(image, dtype=np.float32) / 255.0
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 256.0
                mask_sky = np.asarray(mask_sky)
                depth_gt[mask_sky] = 0
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'diode_train' in image_path:
                image = Image.open(image_path).crop((76, 39, 948, 729)).resize((565, 427), Image.ANTIALIAS)
                image = np.asarray(image, dtype=np.float32) / 255.0
                depth_gt = np.load(depth_path).squeeze()[39:729, 76:948]
                depth_gt = cv2.resize(depth_gt, dsize=(565, 427), interpolation=cv2.INTER_NEAREST)
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'UASOL' in image_path:
                mask_path = sample_path.split()[3]
                image = Image.open(image_path).crop((324, 36, 1884, 1206)).resize((640, 480), Image.ANTIALIAS)
                depth_gt = Image.open(depth_path).crop((324, 36, 1884, 1206)).resize((640, 480), Image.NEAREST)
                mask = Image.open(mask_path).crop((324, 36, 1884, 1206)).resize((640, 480), Image.NEAREST)
                image = np.asarray(image, dtype=np.float32) / 255.0
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 256.0
                mask = np.asarray(mask)
                depth_gt[mask] = 0
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'ApolloScape' in image_path:
                mask_path = sample_path.split()[3]
                image = Image.open(image_path)  # .crop((43, 45, 608, 472))
                depth_gt = Image.open(depth_path)  # .crop((43, 45, 608, 472))
                mask = Image.open(mask_path)  # .crop((43, 45, 608, 472))
                random_angle = (random.random() - 0.5) * 2
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
                mask = self.rotate_image(mask, random_angle, flag=Image.NEAREST)
                image = np.asarray(image, dtype=np.float32) / 255.0
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                mask = np.asarray(mask)
                depth_gt = np.expand_dims(depth_gt, axis=2)
                depth_gt[mask] = 0
                depth_gt = depth_gt / 200.0
            elif 'kitti_dataset' in image_path:
                image = Image.open(image_path).crop((220, 0, 1020, 375)).resize((640, 300), Image.ANTIALIAS)
                depth_gt = Image.open(depth_path).crop((220, 0, 1020, 375)).resize((640, 300), Image.NEAREST)
                image = np.asarray(image, dtype=np.float32) / 255.0
                image = np.pad(image, ((180, 0), (0, 0), (0, 0)), 'constant', constant_values=(1))  
                image = image[46:470, 44:608]    
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 256.0
                depth_gt = np.pad(depth_gt, ((180, 0), (0, 0)), 'constant', constant_values=(0))  
                depth_gt = depth_gt[46:470, 44:608]
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'Argoverse' in image_path:
                image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
                depth_gt = np.asarray(Image.open(depth_path), dtype=np.float32) / 256.0
                image, depth_gt = auto_fov_fitting(image, depth_gt, 1777.100796, 1777.100796)
                depth_gt = cv2.resize(depth_gt, dsize=(564, 424), interpolation=cv2.INTER_NEAREST)
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif '7scenes' in image_path:
                image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
                depth_gt = np.asarray(Image.open(depth_path), dtype=np.float32) / 1000.0
                image, depth_gt = auto_fov_fitting(image, depth_gt, 585, 585)
                depth_gt = cv2.resize(depth_gt, dsize=(564, 424), interpolation=cv2.INTER_NEAREST)
                depth_gt = np.expand_dims(depth_gt, axis=2)
            else:
                image = Image.open(image_path).crop((65, 48, 575, 432)).resize((564, 424), Image.ANTIALIAS)
                depth_gt = Image.open(depth_path).crop((65, 48, 575, 432)).resize((564, 424), Image.NEAREST)
                image = np.asarray(image, dtype=np.float32) / 255.0
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2)
                if 'SUNRGBD' in image_path:
                    depth_gt = depth_gt / 10000.0
                elif 'hua' in image_path:
                    depth_gt = depth_gt / 6553.6
                else:
                    depth_gt = depth_gt / 1000.0

            depth_gt = remove_invalidpix(depth_gt)
            image = image[:,:,:3]
            if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
                image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)

            sample = {'image': image, 'depth': depth_gt, 'image_path': image_path, 
                      'depth_path': depth_path, 'depth_type': depth_type}

        elif self.mode == 'online_eval':
            image_path = sample_path.split()[0]
            depth_path = sample_path.split()[1]
            depth_type = 0

            if 'ApolloScape' in image_path:
                mask_path = sample_path.split()[3]
                image = Image.open(image_path).crop((44, 46, 608, 470))
                depth_gt = Image.open(depth_path).crop((44, 46, 608, 470))
                mask = Image.open(mask_path).crop((44, 46, 608, 470))
                image = np.asarray(image, dtype=np.float32) / 255.0
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                mask = np.asarray(mask)
                depth_gt[mask] = 0
                depth_gt = depth_gt / 200.0
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'nuScenes' in image_path:
                image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
                depth_gt = np.asarray(Image.open(depth_path), dtype=np.float32) / 256.0
                image, depth_gt = auto_fov_fitting(image, depth_gt, 1266.4172, 1266.4172)
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'DIML/indoor' in image_path:
                image = Image.open(image_path).crop((79, 0, 1186, 755)).resize((640, 408), Image.ANTIALIAS)
                depth_gt = Image.open(depth_path).crop((79, 0, 1186, 755)).resize((640, 408), Image.NEAREST)
                image = np.asarray(image, dtype=np.float32) / 255.0
                image = np.pad(image, ((8, 8), (0, 0), (0, 0)), 'constant', constant_values=(0))
                image = image[:, 44:608]
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 1000.0
                depth_gt = np.pad(depth_gt, ((8, 8), (0, 0)), 'constant', constant_values=(0))
                depth_gt = np.expand_dims(depth_gt, axis=2)
                depth_gt = depth_gt[:, 44:608]
            elif 'DIML/outdoor' in image_path:
                image = Image.open(image_path).crop((424, 148, 1496, 932)).resize((640, 480), Image.ANTIALIAS)
                depth_gt = Image.open(depth_path).crop((424, 148, 1496, 932)).resize((640, 480), Image.NEAREST)
                image = np.asarray(image, dtype=np.float32) / 255.0
                image = image[46:470, 44:608]
                depth_gt = np.asarray(Image.open(depth_path), dtype=np.float32) / 1000.0
                depth_gt = np.expand_dims(depth_gt, axis=2)
                depth_gt = depth_gt[46:470, 44:608]
            elif 'diode_val' in image_path:
                image = Image.open(image_path).crop((18, 0, 1006, 783)).resize((640, 480), Image.ANTIALIAS)
                image = np.asarray(image, dtype=np.float32) / 255.0
                image = image[46:470, 44:608]
                depth_gt = np.load(depth_path).squeeze()[:783, 18:1006]
                mask = np.load(depth_path.replace('depth.npy', 'depth_mask.npy'))[:783, 18:1006]
                depth_gt[mask == 0] = 0
                depth_gt = depth_gt[75:767, 68:938]
            elif 'nyu' in image_path:
                image = Image.open(image_path).crop((65, 48, 575, 432)).resize((564, 424), Image.ANTIALIAS)  # eigen_crop
                image = np.asarray(image, dtype=np.float32) / 255.0
                depth_gt = Image.open(depth_path).crop((65, 48, 575, 432))
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 1000.0
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'kitti_dataset' in image_path:
                image = Image.open(image_path).crop((220, 0, 1020, 375)).resize((640, 300), Image.ANTIALIAS)
                image = np.asarray(image, dtype=np.float32) / 255.0
                image = np.pad(image, ((180, 0), (0, 0), (0, 0)), 'constant', constant_values=(1))  
                image = image[46:470, 44:608]    
                depth_gt = Image.open(depth_path).crop((220, 0, 1020, 375))
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 256.0
                depth_gt[:153, :] = 0  # garg_crop
                depth_gt[372:, :] = 0
                depth_gt = np.pad(depth_gt, ((224, 0), (0, 0)), 'constant', constant_values=(0))
                depth_gt = depth_gt[58:588, 55:760]
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'vkitti' in image_path:
                image = Image.open(image_path).crop((221, 0, 1021, 374)).resize((640, 300), Image.ANTIALIAS)
                image = np.asarray(image, dtype=np.float32) / 255.0
                image = np.pad(image, ((180, 0), (0, 0), (0, 0)), 'constant', constant_values=(1))  
                image = image[46:470, 44:608]    
                depth_gt = Image.open(depth_path).crop((221, 0, 1021, 374))
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 100.0
                depth_gt[:153, :] = 0  # garg_crop
                depth_gt[371:, :] = 0
                depth_gt = np.pad(depth_gt, ((224, 0), (0, 0)), 'constant', constant_values=(0))
                depth_gt = depth_gt[58:588, 55:760]
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'DDAD' in image_path:
                image = Image.open(image_path).resize((510, 320), Image.ANTIALIAS)
                image = np.asarray(image, dtype=np.float32) / 255.0
                image = np.pad(image, ((104, 0), (27, 27), (0, 0)), 'constant', constant_values=(1))  
                depth_gt = Image.open(depth_path)
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 256.0
                depth_gt[:496, :] = 0  # garg_crop
                depth_gt[1206:, :] = 0
                depth_gt[:, :69] = 0
                depth_gt[:, 1866:] = 0
                depth_gt = np.pad(depth_gt, ((395, 0), (103, 103)), 'constant', constant_values=(0))
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'TUM' in image_path:
                if 'freiburg1' in image_path:
                    image = Image.open(image_path).crop((66, 49, 574, 431)).resize((564, 424), Image.ANTIALIAS)
                    depth_gt = Image.open(depth_path).crop((66, 49, 574, 431))
                elif 'freiburg2' in image_path:
                    image = Image.open(image_path).crop((64, 47, 576, 433)).resize((564, 424), Image.ANTIALIAS)
                    depth_gt = Image.open(depth_path).crop((64, 47, 576, 433))
                elif 'freiburg3' in image_path:
                    image = Image.open(image_path).crop((57, 41, 583, 439)).resize((564, 424), Image.ANTIALIAS)
                    depth_gt = Image.open(depth_path).crop((57, 41, 583, 439))
                image = np.asarray(image, dtype=np.float32) / 255.0
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 5000.0
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'ibims' in image_path:
                mask_path = depth_path.replace('depth', 'mask_invalid')
                image = Image.open(image_path).crop((51, 52, 601, 464)).resize((564, 424), Image.ANTIALIAS)
                image = np.asarray(image, dtype=np.float32) / 255.0
                depth_gt = Image.open(depth_path).crop((51, 52, 601, 464))
                mask = Image.open(mask_path).crop((51, 52, 601, 464))
                valid_mask = np.asarray(mask, dtype=np.float32) == 1
                depth_gt = np.asarray(depth_gt, dtype=np.float32)*50.0/65535
                depth_gt[~valid_mask] = 0
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'SUNRGBD' in image_path:
                image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
                depth_raw = np.asarray(Image.open(depth_path), dtype=np.int16)
                depth_gt = np.right_shift(depth_raw, 3) | np.left_shift(depth_raw, (16-3))
                depth_gt = depth_gt.astype(np.float32)
                depth_gt /= 1000.0
                if 'kv2/kinect2data' in image_path:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 529.5, 529.5)
                elif 'kv2/align_kv2' in image_path:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 529.5, 529.5)
                elif 'kv1/NYUdata' in image_path:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 518.857901, 519.469611)
                elif 'kv1/b3dodata' in image_path:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 520.532, 520.7444)
                elif 'xtion/sun3ddata' in image_path:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 570.342205, 570.342205)
                elif 'xtion/xtion_align_data' in image_path:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 570.342224, 570.342224)
                elif 'realsense/lg' in image_path:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 693.74469, 693.74469)
                elif 'realsense/sa' in image_path:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 693.74469, 693.74469)
                else:
                    print("Invalid image")
                    assert 1<0
                depth_gt = np.expand_dims(depth_gt, axis=2)
            elif 'ETH3D' in image_path:
                image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
                depth_gt = Image.open(depth_path).resize((image.shape[1], image.shape[0]), Image.NEAREST)
                depth_gt = np.asarray(depth_gt, dtype=np.float32) / 256.0

                depth_gt[:int(0.40810811 * depth_gt.shape[0]), :] = 0
                depth_gt[int(0.99189189 * depth_gt.shape[0]):, :] = 0
                depth_gt[:, :int(0.03594771 * depth_gt.shape[1])] = 0
                depth_gt[:, int(0.96405229 * depth_gt.shape[1]):] = 0
                depth_gt = np.expand_dims(depth_gt, axis=2)
                if image.shape[1] == 6208 and image.shape[0] == 4134:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3408.35, 3408.8)
                elif image.shape[1] == 6208 and image.shape[0] == 4135:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3408.59, 3408.87)
                elif image.shape[1] == 6200 and image.shape[0] == 4131:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3412.23, 3413.08)
                elif image.shape[1] == 6200 and image.shape[0] == 4132:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3412.14, 3413.09)
                elif image.shape[1] == 6200 and image.shape[0] == 4134:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3407.41, 3408.08)
                elif image.shape[1] == 6201 and image.shape[0] == 4130:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3414.66, 3413.37)
                elif image.shape[1] == 6201 and image.shape[0] == 4135:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3406.79, 3404.57)
                elif image.shape[1] == 6203 and image.shape[0] == 4134:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3409.18, 3408.84)
                elif image.shape[1] == 6204 and image.shape[0] == 4132:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3415.73, 3413.89)
                elif image.shape[1] == 6204 and image.shape[0] == 4135:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3410.51, 3409.46)
                elif image.shape[1] == 6205 and image.shape[0] == 4134:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3408.91, 3408.38) 
                elif image.shape[1] == 6205 and image.shape[0] == 4135:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3409.58, 3409.44)
                elif image.shape[1] == 6205 and image.shape[0] == 4136:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3412.13, 3409.71)
                elif image.shape[1] == 6206 and image.shape[0] == 4133:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3410.68, 3409.64)
                elif image.shape[1] == 6209 and image.shape[0] == 4137:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3428.8, 3428.38) 
                elif image.shape[1] == 6198 and image.shape[0] == 4129:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3411.42, 3410.02)
                elif image.shape[1] == 6172 and image.shape[0] == 4118:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3400.23, 3404.6) 
                elif image.shape[1] == 6192 and image.shape[0] == 4121:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3398.23, 3396.19)  
                elif image.shape[1] == 6193 and image.shape[0] == 4127:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3410.3, 3411.7) 
                elif image.shape[1] == 6198 and image.shape[0] == 4130:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3408.95, 3408.5) 
                elif image.shape[1] == 6198 and image.shape[0] == 4132:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3404.55, 3404.11)  
                elif image.shape[1] == 6220 and image.shape[0] == 4141:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3430.27, 3429.23)
                elif image.shape[1] == 6221 and image.shape[0] == 4146:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3437.84, 3435.95) 
                elif image.shape[1] == 6211 and image.shape[0] == 4137:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3410.34, 3409.98) 
                elif image.shape[1] == 6212 and image.shape[0] == 4139:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3413.07, 3413.66)
                elif image.shape[1] == 6212 and image.shape[0] == 4141:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3415.02, 3414.94) 
                elif image.shape[1] == 6214 and image.shape[0] == 4139:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3434.17, 3433.62)
                elif image.shape[1] == 6215 and image.shape[0] == 4147:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3429.76, 3429.06)
                elif image.shape[1] == 6215 and image.shape[0] == 4140:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3432.98, 3431.61)
                elif image.shape[1] == 6215 and image.shape[0] == 4141:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3433.17, 3432.1)
                elif image.shape[1] == 6221 and image.shape[0] == 4146:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3431.61, 3429.47)
                elif image.shape[1] == 6223 and image.shape[0] == 4148:
                    image, depth_gt = auto_fov_fitting(image, depth_gt, 3422.09, 3419.97)
                else:
                    print(depth_path, depth_gt.shape[:2])
                    assert 1<0
                
            elif 'buptdepth' in image_path:
                image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
                depth_gt = np.asarray(Image.open(depth_path), dtype=np.float32) * 1.2231 / 256.0
                assert len(sample_path.split()) > 2
                mask_path = sample_path.split()[-1]
                mask = np.asarray(Image.open(mask_path))
                depth_gt[mask] = 0
                image, depth_gt = auto_fov_fitting(image, depth_gt, 1091.51752, 1091.51752)
            else:
                print("Invalid dataset")
                exit
            
            sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': True, 
                      'image_path': image_path, 'depth_path': depth_path,
                      'depth_type': depth_type}
        else:
            image_path = sample_path.split()[0].strip()
            save_depth_path = sample_path.split()[1].strip()
            fx = int(float(sample_path.split()[-2]))
            fy = int(float(sample_path.split()[-1]))
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
            image, image_origin_size = auto_fov_fitting_test(image, fx, fy)
            
            # only use "image_origin_size" to record image's original size
            sample = {'image': image, 'image_path': image_path, 'depth': image_origin_size, 'depth_path': save_depth_path, 'depth_type': 0}
        

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'multi':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, image_path = sample['image'], sample['image_path']
        depth_path, depth_type = sample['depth_path'], sample['depth_type']
        image = self.to_tensor(image)
        image = self.normalize(image)

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'image_path': image_path, 'depth_type': depth_type}
        elif self.mode == 'online_eval':
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'has_valid_depth': has_valid_depth,
                    'image_path': image_path, 'depth_path': depth_path,
                    'depth_type': depth_type}
        else:
            return {'image': image, 'depth': depth, 'save_depth_path': depth_path}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


def remove_invalidpix(depth_gt):
    depth_gt[depth_gt > 80.0] = 0
    depth_gt[np.isinf(depth_gt)] = 0
    depth_gt[np.isnan(depth_gt)] = 0

    return depth_gt
