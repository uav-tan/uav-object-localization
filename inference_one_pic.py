import argparse
from pathlib import Path
import glob
from PIL import Image
import ttach as tta
import cv2
import numpy as np
import torch
import albumentations as albu

from cfg import py2cfg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_supervision import *
import random
import os


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def pv2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def landcoverai_to_rgb(mask):
    w, h = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(w, h, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [233, 193, 133]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 255, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def uavid2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 64, 128]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 128, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 128, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [64, 0, 128]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [192, 0, 192]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [64, 64, 0]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 0, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--image_path", type=str, default='../master/seq48_ground_truth/build/0000000001.jpg', help="Path to  huge image")#'data/uavid/uavid_test'
    arg("-c", "--config_path", type=Path, default='./uavid/unetformer.py', help="Path to  config")
    arg("-o", "--output_path", type=Path, default='../master/seq48_ground_truth/mask/npy', help="Path to save resulting masks.")#'fig_results/uavid/unetformer_r18'
    arg("-t", "--tta", help="Test time augmentation.", default="lr", choices=[None, "d4", "lr"])
    arg("-ph", "--patch-height", help="height of patch size", type=int, default=1152)#1152
    arg("-pw", "--patch-width", help="width of patch size", type=int, default=1024)#1024
    arg("-b", "--batch-size", help="batch size", type=int, default=4)
    arg("-d", "--dataset", help="dataset", default="uavid", choices=["pv", "landcoverai", "uavid"])
    arg("--read_from", help="read test data from ", default="path", choices=["data", "path", "txt"])
    arg("--generate_dydepth_npy", help=" generate dynamic_object_mask_npy for dynamic_depth", default=False)
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path, model):
    pretrained_dict = torch.load(checkpoint_path)['model_state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]

    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
    # print(oh, ow, rh, rw, height_pad, width_pad)
    h, w = oh + height_pad, ow + width_pad

    pad = albu.PadIfNeeded(min_height=h, min_width=w, border_mode=0,
                           position='bottom_right', value=[0, 0, 0])(image=image)
    img_pad = pad['image']
    return img_pad, height_pad, width_pad


class InferenceDataset(Dataset):
    def __init__(self, tile_list=None, transform=albu.Normalize()):
        self.tile_list = tile_list
        self.transform = transform

    def __getitem__(self, index):
        img = self.tile_list[index]
        img_id = index
        aug = self.transform(image=img)
        img = aug['image']
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        results = dict(img_id=img_id, img=img)
        return results

    def __len__(self):
        return len(self.tile_list)


def make_dataset_for_one_huge_image(img_path, patch_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tile_list = []
    image_pad, height_pad, width_pad = get_img_padded(img.copy(), patch_size)

    output_height, output_width = image_pad.shape[0], image_pad.shape[1]

    for x in range(0, output_height, patch_size[0]):
        for y in range(0, output_width, patch_size[1]):
            image_tile = image_pad[x:x+patch_size[0], y:y+patch_size[1]]
            tile_list.append(image_tile)

    dataset = InferenceDataset(tile_list=tile_list)
    return dataset, width_pad, height_pad, output_width, output_height, image_pad, img.shape


def GeoSeg_mask(img_paths):
    args = get_args()
    seed_everything(42)
    # seqs = os.listdir(args.image_path)

    # print(img_paths)
    patch_size = (args.patch_height, args.patch_width)
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)

    model.cuda(config.gpus[0])
    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1, 1.25, 1.5, 1.75]),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)


    # if (args.read_from == "data"):
    #     for ext in ('*.tif', '*.png', '*.jpg'): #从数据文件夹中读取数据
    #         img_paths.extend(glob.glob(os.path.join(args.image_path,  'Images', ext)))#str(seq),
    # if (args.read_from == "path"):#从根文件夹中读取数据
    #     img_files = image_path #'data'
    #     for img_file in img_files:
    #         img_paths.extend(glob.glob(os.path.join(args.image_path, img_file)))#'data'
    # if (args.read_from == "txt"):
    #     model_name = 'test_files'
    #     split_txt = "../Monocular-UAV-videos/splits/uav_yang_add/{}.txt".format(model_name)
    #     f = open(split_txt, 'r')
    #     dta = f.readlines()
    #     for dd in dta:
    #         img_paths.extend(glob.glob(os.path.join(args.image_path, dd.split()[0], dd.split()[1]+'.jpg')))


    img_paths.sort()
    # print(img_paths)
    for img_path in tqdm(img_paths):
        img_name = img_path.split('/')[-1]
        # print('origin mask', original_mask.shape)
        if not args.generate_dydepth_npy:
            output_path = os.path.join(args.output_path, 'Labels')
        else:
            output_path = os.path.join(args.output_path, img_path.split('/')[-4])
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if (args.generate_dydepth_npy):
            idx = [-1, 0, 1]
    
        else:
            idx = [0]
        for ids in idx:
            img_str_before = img_name.split('.')[0]
            img_str_atr = str(int(img_str_before) + ids).zfill(10)
            img_path_changed = img_path.replace(img_str_before, img_str_atr)
            dataset, width_pad, height_pad, output_width, output_height, img_pad, img_shape = \
                make_dataset_for_one_huge_image(img_path_changed, patch_size)
            # print('img_padded', img_pad.shape)
            output_mask = np.zeros(shape=(output_height, output_width), dtype=np.uint8)
            output_tiles = []
            k = 0
            with torch.no_grad():
                dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                        drop_last=False, shuffle=False)
                for input in dataloader: #tqdm
                    # raw_prediction NxCxHxW
                    raw_predictions = model(input['img'].cuda(config.gpus[0]))
                    # print('raw_pred shape:', raw_predictions.shape)
                    raw_predictions = nn.Softmax(dim=1)(raw_predictions)
                    # input_images['features'] NxCxHxW C=3
                    predictions = raw_predictions.argmax(dim=1)
                    image_ids = input['img_id']
                    # print('prediction', predictions.shape)
                    # print(np.unique(predictions))

                    for i in range(predictions.shape[0]):
                        raw_mask = predictions[i].cpu().numpy()
                        mask = raw_mask
                        output_tiles.append((mask, image_ids[i].cpu().numpy()))

            for m in range(0, output_height, patch_size[0]):
                for n in range(0, output_width, patch_size[1]):
                    output_mask[m:m + patch_size[0], n:n + patch_size[1]] = output_tiles[k][0]
                    k = k + 1

            output_mask = output_mask[-img_shape[0]:, -img_shape[1]:]
            #输出的就是掩膜，此处生成npy
            
            # out_dynamic_mask_path = '../DynamicDepth/train_mask/npy'
            obj_mask = np.where(output_mask == 1, True, False) #Moving_Car(4) road(1)
            if (ids == -1):
                np.save(os.path.join(output_path, '{}-1.npy'.format(img_str_before)), obj_mask)
            if (ids == 0):
                np.save(os.path.join(output_path, '{}.npy'.format(img_str_before)), obj_mask)
            if (ids == 1):
                np.save(os.path.join(output_path, '{}+1.npy'.format(img_str_before)), obj_mask)
                # output_path = '../DynamicDepth/train_mask/img'
            # # print('mask', output_mask.shape)
            # # if args.dataset == 'landcoverai':
            # #     output_mask = landcoverai_to_rgb(output_mask)
            # # elif args.dataset == 'pv':
            # #     output_mask = pv2rgb(output_mask)
            # # elif args.dataset == 'uavid':
            # output_mask_color = uavid2rgb(output_mask)
            # # else:
            # #     output_mask = output_mask
            # # assert img_shape == output_mask.shape
            # cv2.imwrite(os.path.join(output_path, img_name), output_mask_color)
    return output_mask  

if __name__ == "__main__":
    img_paths = ['../DNet-master/seq48_ground_truth/build/0000000001.jpg']
    GeoSeg_mask(img_paths)
