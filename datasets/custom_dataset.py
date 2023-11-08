import torch.utils.data as data
import os
from PIL import Image
import torchvision.transforms as transforms
# import random
# random.seed(42)
import torch
import numpy as np
import glob


class DavisTrainDataset(data.Dataset):
    def __init__(self, data_dir, resolution, size_divisibility, filtering=False):
        super().__init__()
        self.data_dir = data_dir     # [img_dir, pseudo_gt_dir, sem_seg_dir]
        self.img_dir, self.pseudo_gt_dir, self.sem_seg_dir = self.data_dir[0], self.data_dir[1], self.data_dir[2]

        self.target_h, self.target_w = resolution
        self.size_divisibility = size_divisibility

        self.image_path = []
        self.pseudo_gt_path = []
        self.sem_seg_ori_path = []

        fg_thresh = 0.5    # values over this are considered foreground in the sigmoid mask
        conf_thresh = 0.9  # confidence threshold      
        
        train_videos = os.listdir(self.img_dir)
        train_videos = list(filter(lambda x: not x.startswith('.'), train_videos))
        for train_vid in train_videos:
            train_vid_path = os.path.join(self.img_dir, train_vid)
            train_files = sorted(os.listdir(train_vid_path))

            for train_file in train_files:
                train_file_path = os.path.join(train_vid_path, train_file)

                dirname = train_file_path.split('/')[-2]     # bear
                fname = train_file_path.split('/')[-1].split('.')[0]     # 00000
                path_to_pseudo_gt = f"{self.pseudo_gt_dir}/{dirname}/{fname}.png"
                if filtering:
                    # filter the pseudo ground-truth based on confidence threshold
                    im = Image.open(path_to_pseudo_gt)
                    im_np = np.array(im)/255.
                    fg_count = np.count_nonzero(im_np > fg_thresh)
                    im_thresh = im_np > fg_thresh
                    im_over_fg_thresh = np.where(im_thresh.astype(float) == 0, 0, im_np)
                    fg_sum = np.sum(im_over_fg_thresh)

                    ratio = fg_sum/fg_count
                    if ratio > conf_thresh:
                        path_to_binary_dir = path_to_pseudo_gt.replace('sigmoid', 'binary')
                        self.pseudo_gt_path.append(path_to_binary_dir)
                    else:
                        continue
                else:
                    path_to_binary_dir = path_to_pseudo_gt.replace('sigmoid', 'binary')
                    self.pseudo_gt_path.append(path_to_binary_dir)

                self.image_path.append(train_file_path)      # ../DAVIS/JPEGImages/480p/bear/00000.jpg

                path_to_sem_seg_ori = f"{self.sem_seg_dir}/{dirname}/{fname}.png"
                self.sem_seg_ori_path.append(path_to_sem_seg_ori)


        assert (len(self.image_path) == len(self.pseudo_gt_path)) and (len(self.image_path) == len(self.sem_seg_ori_path)), "Sanity check failed. images and masks in unequal numbers!"

        self.transform_image = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.transform_pseudo_gt = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.NEAREST), transforms.ToTensor()]) 
        self.transform_sem_seg = transforms.ToTensor()     


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        sample = {}
        img_path = self.image_path[index]
        pseudo_gt_path = self.pseudo_gt_path[index]
        sem_seg_ori_path = self.sem_seg_ori_path[index]

        sample["image"] = self.transform_image(Image.open(img_path))
        sample["pseudo_gt"] = self.transform_pseudo_gt(Image.open(pseudo_gt_path))    # graphcut masks
        sample["sem_seg_ori"] = self.transform_sem_seg(Image.open(sem_seg_ori_path))  # ground truth

        dirname = img_path.split('/')[-2]
        fname = img_path.split('/')[-1].split('.')[0]
        sample["dirname"] = dirname
        sample["fname"] = fname

        return sample



class DavisValDataset(data.Dataset):
    def __init__(self, data_dir, resolution, size_divisibility):
        super().__init__()
        self.data_dir = data_dir     # [img_dir, pseudo_gt_dir, sem_seg_dir]
        self.img_dir, self.pseudo_gt_dir, self.sem_seg_dir = self.data_dir[0], self.data_dir[1], self.data_dir[2]

        self.target_h, self.target_w = resolution
        self.size_divisibility = size_divisibility

        self.image_path = []
        self.pseudo_gt_path = []
        self.sem_seg_ori_path = []
        
        val_videos = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 
                      'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf',
                      'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']
        for val_vid in val_videos:
            val_vid_path = os.path.join(self.img_dir, val_vid)
            val_files = sorted(os.listdir(val_vid_path))[1:]       # drop the first frame -- for evaluation

            for val_file in val_files:
                val_file_path = os.path.join(val_vid_path, val_file)
                self.image_path.append(val_file_path)      # ../DAVIS/JPEGImages/480p/bear/00000.jpg

                dirname = val_file_path.split('/')[-2]     # bear
                fname = val_file_path.split('/')[-1].split('.')[0]     # 00000
                path_to_pseudo_gt = f"{self.pseudo_gt_dir}/{dirname}/{fname}.png"
                self.pseudo_gt_path.append(path_to_pseudo_gt)

                path_to_sem_seg_ori = f"{self.sem_seg_dir}/{dirname}/{fname}.png"
                self.sem_seg_ori_path.append(path_to_sem_seg_ori)

        assert (len(self.image_path) == len(self.pseudo_gt_path)) and (len(self.image_path) == len(self.sem_seg_ori_path)), "Sanity check failed. images and masks in unequal numbers!"

        self.transform_image = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.transform_pseudo_gt = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.NEAREST), transforms.ToTensor()]) 
        self.transform_sem_seg = transforms.ToTensor()     


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        sample = {}

        img_path = self.image_path[index]
        dirname = img_path.split('/')[-2]
        fname = img_path.split('/')[-1].split('.')[0]

        pseudo_gt_path = self.pseudo_gt_path[index]
        sem_seg_ori_path = self.sem_seg_ori_path[index]

        sample["image"] = self.transform_image(Image.open(img_path))
        sample["pseudo_gt"] = self.transform_pseudo_gt(Image.open(pseudo_gt_path))    # graphcut masks
        sample["sem_seg_ori"] = self.transform_sem_seg(Image.open(sem_seg_ori_path))  # ground truth
        sample["dirname"] = dirname
        sample["fname"] = fname

        return sample
    
    
    
class STv2Dataset(data.Dataset):
    def __init__(self, data_dir, resolution, size_divisibility, filtering=False):
        super().__init__()
        self.data_dir = data_dir     # [img_dir, pseudo_gt_dir, sem_seg_dir]
        self.img_dir, self.pseudo_gt_dir, self.sem_seg_dir = self.data_dir[0], self.data_dir[1], self.data_dir[2]

        self.target_h, self.target_w = resolution
        self.size_divisibility = size_divisibility

        self.image_path = []
        self.pseudo_gt_path = []
        self.sem_seg_ori_path = []
          
        train_videos = glob.glob(f'{self.img_dir}/*')
        train_videos = list(filter(lambda x: os.path.isdir(x), train_videos))
        
        for train_vid_path in train_videos:
            video_name = train_vid_path.split('/')[-1]
            train_files = os.listdir(train_vid_path)
            train_files = list(filter(lambda x: not x.startswith('.'), train_files))

            for train_file in train_files:
                if video_name in ["penguin", "monkeydog"]:
                    sem_seg_file = f"{train_file.split('.')[0]}.png"
                    pseudo_gt_file = sem_seg_file
                elif video_name in ["girl", "cheetah"]:
                    sem_seg_file = train_file
                    pseudo_gt_file = f"{train_file.split('.')[0]}.png"    
                else:
                    sem_seg_file = train_file
                    pseudo_gt_file = train_file
                    
                train_file_path = os.path.join(train_vid_path, train_file)
                self.image_path.append(train_file_path)

                dirname = train_file_path.split('/')[-2]     # bear
                
                path_to_pseudo_gt = os.path.join(self.pseudo_gt_dir, dirname, pseudo_gt_file)
                self.pseudo_gt_path.append(path_to_pseudo_gt)

                path_to_sem_seg_ori = os.path.join(self.sem_seg_dir, dirname, sem_seg_file)
                self.sem_seg_ori_path.append(path_to_sem_seg_ori)


        assert (len(self.image_path) == len(self.pseudo_gt_path)) and (len(self.image_path) == len(self.sem_seg_ori_path)), "Sanity check failed. images and masks in unequal numbers!"

        self.transform_image = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.transform_pseudo_gt = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.NEAREST), transforms.ToTensor()]) 
        self.transform_sem_seg = transforms.ToTensor()     


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        sample = {}
        img_path = self.image_path[index]
        pseudo_gt_path = self.pseudo_gt_path[index]
        sem_seg_ori_path = self.sem_seg_ori_path[index]

        sample["image"] = self.transform_image(Image.open(img_path))
        sample["pseudo_gt"] = self.transform_pseudo_gt(Image.open(pseudo_gt_path))       # graphcut masks
        sample["sem_seg_ori"] = self.transform_sem_seg(Image.open(sem_seg_ori_path))[0].unsqueeze(0)  # 3xHxW -> HxW; actual ground truth

        dirname = sem_seg_ori_path.split('/')[-2]
        fname = pseudo_gt_path.split('/')[-1]
        sample["dirname"] = dirname
        sample["fname"] = fname

        return sample



class FBMSTrainDataset(data.Dataset):
    def __init__(self, data_dir, resolution, size_divisibility, filtering=False):
        super().__init__()
        self.data_dir = data_dir     # [img_dir, pseudo_gt_dir, sem_seg_dir]
        self.img_dir, self.pseudo_gt_dir, self.sem_seg_dir = self.data_dir[0], self.data_dir[1], self.data_dir[2]

        self.target_h, self.target_w = resolution
        self.size_divisibility = size_divisibility

        self.image_path = []
        self.pseudo_gt_path = []
        self.sem_seg_ori_path = []

        # self.img_dir: /path/to/FBMS59
        train_videos_path = [os.path.join(self.img_dir, 'Trainingset', vid_name) for vid_name in os.listdir(os.path.join(self.img_dir, 'Trainingset'))]
        train_videos_path += [os.path.join(self.img_dir, 'Testset', vid_name) for vid_name in os.listdir(os.path.join(self.img_dir, 'Testset'))]
        train_videos_path = list(filter(lambda x: os.path.isdir(x), train_videos_path))

        for train_video_path in train_videos_path:
            # train_video = train_video.strip()  # camel01
            # train_vid_path = os.path.join(self.img_dir, train_video)
            train_files = os.listdir(train_video_path)
            train_files = list(filter(lambda x: x.endswith('.jpg'), train_files))
            vid_name = train_video_path.split('/')[-1]

            for train_file in train_files:
                if vid_name == "tennis":
                    fnum = int(train_file.split('.')[0][6:])
                else:
                    fnum = int(train_file.split('.')[0].split('_')[-1])

                pseudo_gt_fname = f"{str(fnum).zfill(5)}.png"
                path_to_pseudo_gt = f"{self.pseudo_gt_dir}/{vid_name}/{train_file.replace('.jpg', '.png')}"
                self.pseudo_gt_path.append(path_to_pseudo_gt)

                train_file_path = os.path.join(train_video_path, train_file)
                self.image_path.append(train_file_path) 

                self.sem_seg_ori_path.append(path_to_pseudo_gt)     


        assert (len(self.image_path) == len(self.pseudo_gt_path)), "Sanity check failed. images and masks in unequal numbers!"

        self.transform_image = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.transform_pseudo_gt = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.NEAREST), transforms.ToTensor()]) 
        self.transform_sem_seg = transforms.ToTensor()   


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        sample = {}
        img_path = self.image_path[index]
        pseudo_gt_path = self.pseudo_gt_path[index]
        # sem_seg_ori_path = self.sem_seg_ori_path[index]

        sample["image"] = self.transform_image(Image.open(img_path))
        sample["pseudo_gt"] = self.transform_pseudo_gt(Image.open(pseudo_gt_path))    # graphcut masks
        sample["sem_seg_ori"] = self.transform_sem_seg(Image.open(img_path))  # ground truth

        dirname = pseudo_gt_path.split('/')[-2]
        fname = pseudo_gt_path.split('/')[-1]
        sample["dirname"] = dirname
        sample["fname"] = fname

        return sample


    
class FBMSValDataset(data.Dataset):
    def __init__(self, data_dir, resolution, size_divisibility, filtering=False):
        super().__init__()
        self.data_dir = data_dir     # [img_dir, pseudo_gt_dir, sem_seg_dir]
        self.img_dir, self.pseudo_gt_dir, self.sem_seg_dir = self.data_dir[0], self.data_dir[1], self.data_dir[2]

        self.target_h, self.target_w = resolution
        self.size_divisibility = size_divisibility

        self.image_path = []
        self.sem_seg_ori_path = []
        self.pseudo_gt_path = []
          
        with open('../../FBMS59_clean/test_dirs.txt', 'r') as fp:
            val_videos = fp.readlines()

        for val_video in val_videos:
            val_video = val_video.strip()  # camel01

            val_vid_path = os.path.join(self.img_dir, val_video)
            val_files = os.listdir(val_vid_path)
            val_files = list(filter(lambda x: not x.startswith('.'), val_files))

            for val_file in val_files:
                gt_file = f"{val_file.split('.')[0]}.png"
                val_file_path = os.path.join(val_vid_path, val_file)
                
#                 path_to_pseudo_gt = f"{self.pseudo_gt_dir}/{dirname}/{fname}.png"
                self.image_path.append(val_file_path)      

                path_to_sem_seg_ori = os.path.join(self.sem_seg_dir, val_video, gt_file)
                self.sem_seg_ori_path.append(path_to_sem_seg_ori)
                self.pseudo_gt_path.append(path_to_sem_seg_ori)


        assert (len(self.image_path) == len(self.sem_seg_ori_path)), "Sanity check failed. images and masks in unequal numbers!"
#         (len(self.image_path) == len(self.pseudo_gt_path)) and

        self.transform_image = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.transform_pseudo_gt = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.NEAREST), transforms.ToTensor()]) 
        self.transform_sem_seg = transforms.ToTensor()     


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        sample = {}
        img_path = self.image_path[index]
#         pseudo_gt_path = self.pseudo_gt_path[index]
        sem_seg_ori_path = self.sem_seg_ori_path[index]

        sample["image"] = self.transform_image(Image.open(img_path))
#         sample["pseudo_gt"] = self.transform_pseudo_gt(Image.open(pseudo_gt_path))    # graphcut masks
        sample["sem_seg_ori"] = self.transform_sem_seg(Image.open(sem_seg_ori_path))  # ground truth

        dirname = sem_seg_ori_path.split('/')[-2]
        fname = sem_seg_ori_path.split('/')[-1]
        sample["dirname"] = dirname
        sample["fname"] = fname

        return sample



class AllVideoTrainDataset(data.Dataset):
    def __init__(self, data_dirs, resolution, size_divisibility, filtering=False):
        super().__init__()

        self.target_h, self.target_w = resolution
        self.size_divisibility = size_divisibility

        davis_data_dir, stv2_data_dir, fbms_data_dir = data_dirs[0], data_dirs[1], data_dirs[2]    # data_dirs: [[],[],[]]
        
        self.images_path = []
        self.pseudo_gts_path = []

        self.davis_train_dataset = DavisTrainDataset(davis_data_dir, resolution, size_divisibility)
        self.stv2_train_dataset = STv2Dataset(stv2_data_dir, resolution, size_divisibility)
        self.fbms_train_dataset = FBMSTrainDataset(fbms_data_dir, resolution, size_divisibility)

        self.len_davis_dataset = self.davis_train_dataset.__len__()
        self.len_stv2_dataset = self.stv2_train_dataset.__len__()
        self.len_fbms_dataset = self.fbms_train_dataset.__len__()


    def __len__(self):
        return self.len_davis_dataset + self.len_stv2_dataset + self.len_fbms_dataset


    def __getitem__(self, index):
        if index < self.len_davis_dataset:
            return self.davis_train_dataset.__getitem__(index)
        
        elif (index >= self.len_davis_dataset) and (index < (self.len_davis_dataset + self.len_stv2_dataset)):
            return self.stv2_train_dataset.__getitem__(index - self.len_davis_dataset)
        
        else:
            return self.fbms_train_dataset.__getitem__(index - (self.len_davis_dataset + self.len_stv2_dataset))


    
class CUBTest(data.Dataset):
    def __init__(self, data_dir, resolution, size_divisibility):
        super().__init__()
        image_dir, sem_seg_dir = data_dir[0], data_dir[1]
        self.target_h, self.target_w = resolution
        self.size_divisibility = size_divisibility

        self.image_path = []
        self.sem_seg_path = []

        test_dirs = glob.glob(f'{image_dir}/*')
        for test_dir in test_dirs:
            dirname = test_dir.split('/')[-1]
            test_files = os.listdir(test_dir)
            test_files = list(filter(lambda x: not x.startswith('.'), test_files))
            for test_file in test_files:
                path_to_image = os.path.join(test_dir, test_file)
                sem_seg_fname = test_file.replace('.jpg', '.png')
                path_to_sem_seg = f'{sem_seg_dir}/{dirname}/{sem_seg_fname}'

                self.image_path.append(path_to_image)
                self.sem_seg_path.append(path_to_sem_seg)

        assert len(self.image_path) == len(self.sem_seg_path), "image and segmentations in unequal numbers!"

        self.transform_image = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.transform_sem_seg = transforms.ToTensor()  


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        sample = {}
        sample["image"] = self.transform_image(Image.open(self.image_path[index]).convert('RGB'))
        sample["sem_seg_ori"] = self.transform_sem_seg(Image.open(self.sem_seg_path[index]))
        dirname = self.sem_seg_path[index].split('/')[-2]
        fname = self.sem_seg_path[index].split('/')[-1]

        sample["dirname"] = dirname
        sample["fname"] = fname
        return sample

        

class DUTSTest(data.Dataset):
    def __init__(self, data_dir, resolution, size_divisibility):
        super().__init__()
        image_dir, sem_seg_dir = data_dir[0], data_dir[1]
        self.target_h, self.target_w = resolution
        self.size_divisibility = size_divisibility

        self.image_path = []
        self.sem_seg_path = []

        test_images = os.listdir(image_dir)
        for test_img in test_images:
            path_to_image = os.path.join(image_dir, test_img)
            sem_seg_fname = test_img.replace('.jpg', '.png')
            path_to_sem_seg = os.path.join(sem_seg_dir, sem_seg_fname)

            self.image_path.append(path_to_image)
            self.sem_seg_path.append(path_to_sem_seg)

        assert len(self.image_path) == len(self.sem_seg_path), "image and segmentations in unequal numbers!"

        self.transform_image = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.transform_sem_seg = transforms.ToTensor()  


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        sample = {}
        sample["image"] = self.transform_image(Image.open(self.image_path[index]).convert('RGB'))
        sample["sem_seg_ori"] = self.transform_sem_seg(Image.open(self.sem_seg_path[index]))
        # dirname = self.sem_seg_path[index].split('/')[-2]
        fname = self.sem_seg_path[index].split('/')[-1]

        # sample["dirname"] = dirname
        sample["fname"] = fname
        return sample



class ECSSDTest(data.Dataset):
    def __init__(self, data_dir, resolution, size_divisibility):
        super().__init__()
        image_dir, sem_seg_dir = data_dir[0], data_dir[1]
        self.target_h, self.target_w = resolution
        self.size_divisibility = size_divisibility

        self.image_path = []
        self.sem_seg_path = []

        test_images = os.listdir(image_dir)
        for test_img in test_images:
            path_to_image = os.path.join(image_dir, test_img)
            sem_seg_fname = test_img.replace('.jpg', '.png')
            path_to_sem_seg = os.path.join(sem_seg_dir, sem_seg_fname)

            self.image_path.append(path_to_image)
            self.sem_seg_path.append(path_to_sem_seg)

        assert len(self.image_path) == len(self.sem_seg_path), "image and segmentations in unequal numbers!"

        self.transform_image = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.transform_sem_seg = transforms.ToTensor()  


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        sample = {}
        sample["image"] = self.transform_image(Image.open(self.image_path[index]).convert('RGB'))
        sample["sem_seg_ori"] = self.transform_sem_seg(Image.open(self.sem_seg_path[index]))

        sample["fname"] = self.sem_seg_path[index].split('/')[-1]
        return sample



class OMRONTest(data.Dataset):
    def __init__(self, data_dir, resolution, size_divisibility):
        super().__init__()
        image_dir, sem_seg_dir = data_dir[0], data_dir[1]
        self.target_h, self.target_w = resolution
        self.size_divisibility = size_divisibility

        self.image_path = []
        self.sem_seg_path = []

        test_images = os.listdir(image_dir)
        for test_img in test_images:
            path_to_image = os.path.join(image_dir, test_img)
            sem_seg_fname = test_img.replace('.jpg', '.png')
            path_to_sem_seg = os.path.join(sem_seg_dir, sem_seg_fname)

            self.image_path.append(path_to_image)
            self.sem_seg_path.append(path_to_sem_seg)

        assert len(self.image_path) == len(self.sem_seg_path), "image and segmentations in unequal numbers!"

        self.transform_image = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.transform_sem_seg = transforms.ToTensor()  


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        sample = {}
        sample["image"] = self.transform_image(Image.open(self.image_path[index]).convert('RGB'))
        sample["sem_seg_ori"] = self.transform_sem_seg(Image.open(self.sem_seg_path[index]))

        sample["fname"] = self.sem_seg_path[index].split('/')[-1]
        return sample



class FlowersTest(data.Dataset):
    def __init__(self, data_dir, resolution, size_divisibility):
        super().__init__()
        image_dir, sem_seg_dir = data_dir[0], data_dir[1]
        self.target_h, self.target_w = resolution
        self.size_divisibility = size_divisibility

        self.image_path = []
        self.sem_seg_path = []

        test_images = os.listdir(image_dir)
        for test_img in test_images:      # image_xxxxx.jpg
            path_to_image = os.path.join(image_dir, test_img)
            sem_seg_fname = test_img.replace('.jpg', '.png')
            sem_seg_fname = f'seg_{sem_seg_fname[6:]}'  # sem_seg_fname.replace('image', 'seg')   # seg_xxxxx.png
            path_to_sem_seg = os.path.join(sem_seg_dir, sem_seg_fname)

            self.image_path.append(path_to_image)
            self.sem_seg_path.append(path_to_sem_seg)

        assert len(self.image_path) == len(self.sem_seg_path), "image and segmentations in unequal numbers!"

        self.transform_image = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.transform_sem_seg = transforms.ToTensor()  


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        sample = {}
        sample["image"] = self.transform_image(Image.open(self.image_path[index]).convert('RGB'))
        sample["sem_seg_ori"] = self.transform_sem_seg(Image.open(self.sem_seg_path[index]))
        sample["fname"] = self.sem_seg_path[index].split('_')[-1]
        return sample



class RandomInternetImages(data.Dataset):
    def __init__(self, data_dir, resolution, size_divisibility):
        super().__init__()
        image_dir = data_dir[0]
        self.target_h, self.target_w = resolution
        self.size_divisibility = size_divisibility

        self.image_path = []

        test_images = os.listdir(image_dir)
        test_images = list(filter(lambda x: not x.startswith('.'), test_images))
        for test_img in test_images:      # image_xxxxx.jpg
            if os.path.isfile(os.path.join(image_dir, test_img)):
                path_to_image = os.path.join(image_dir, test_img)
                self.image_path.append(path_to_image)

        self.transform_image = transforms.Compose([transforms.Resize((self.target_h, self.target_w), Image.Resampling.BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])]) 
        # self.transform_image_ori = transforms.ToTensor()

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        sample = {}
        img = Image.open(self.image_path[index]).convert('RGB')
        orig_w, orig_h = img.size
        sample["image"] = self.transform_image(img)
        sample["orig_size"] = (orig_h, orig_w)
        sample["fname"] = self.image_path[index].split('/')[-1].replace('.jpeg', '.png')
        return sample

