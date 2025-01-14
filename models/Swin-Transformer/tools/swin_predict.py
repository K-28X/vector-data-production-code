import glob
import multiprocessing
import os
from pathlib import Path

import cv2
import torch
from osgeo import gdal
from torch.nn import functional as F
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from matplotlib import pyplot as plt
import mmcv
from collections import Counter
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
# config_file = r'../configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py'
# checkpoint_file = r"F:\lsy\Swin-Transformer-Semantic-Segmentation-main\checkpoints\iter_160000.pth"  # 修改此处权重名，即.pth文件
#
# model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
#
# img_root = Path("F:\lsy\Swin-Transformer-Semantic-Segmentation-main\data\dushangang_test") # 修改要预测的图片名或文件夹名
# save_mask_root = r"F:\lsy\Swin-Transformer-Semantic-Segmentation-main\outputs\dushangang\2"  # 预测结果存放处
# if not os.path.exists(save_mask_root):
#     os.mkdir(save_mask_root)
# img_names = [file for file in os.listdir(img_root) if file.endswith('.tif')]
# for img_name in tqdm(img_names) :
#     # test a single image
#     #img = img_root + img_name
#     print(img_name)
#     img=os.path.join(img_root,img_name)
#     result = inference_segmentor(model, img)[0]
#     img = Image.fromarray(np.uint8(result * 255))
#     img.save(os.path.join(save_mask_root,img_name))


def transformN(img):
    img = img / 255.0

    mean_bgr = np.array([0.485, 0.456, 0.406])
    std_bgr = np.array([0.229, 0.224, 0.225])
    img = np.array(img, dtype=np.float32)
    img -= mean_bgr
    img /= std_bgr
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.copy()).float()
    return img


def tes(model, args):
    input_paths = glob.glob(os.path.join(args.input_path, '*.{}'.format(args.ext)))
    print(args.ext)
    print(input_paths)
    print(len(input_paths))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for input_path in tqdm(input_paths):
        test_path = input_path
        # test_path= test_path.encode("utf8")
        # gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        test_img_gdal = gdal.Open(test_path)
        model = model.cuda()
        model.eval()
        clip_size_r = 512  # 128#256#512#1024
        pad_num = clip_size_r // 4
        clip_size_w = clip_size_r - pad_num  # 256
        cols = test_img_gdal.RasterXSize
        rows = test_img_gdal.RasterYSize
        rows_iters = rows // clip_size_w + 1
        cols_iters = cols // clip_size_w + 1
        print(rows, cols)
        print(rows_iters, cols_iters)
        geo = test_img_gdal.GetGeoTransform()
        proj = test_img_gdal.GetProjection()
        outpath = os.path.join(args.output_path, os.path.basename(test_path))
        out = gdal.GetDriverByName("GTiff").Create(outpath, cols, rows, 1, gdal.GDT_Byte)
        print(outpath)
        out.SetGeoTransform(geo)
        out.SetProjection(proj)
        with torch.no_grad():
            for i in tqdm(range(rows_iters)):
                for j in range(cols_iters):
                    w_y = i * clip_size_w
                    w_x = j * clip_size_w
                    w_endY = (i + 1) * clip_size_w
                    w_endX = (j + 1) * clip_size_w
                    w_sizeX = clip_size_w
                    w_sizeY = clip_size_w

                    pad_top = 0
                    pad_left = 0
                    pad_bottom = 0
                    pad_right = 0

                    r_endY = w_endY + pad_num
                    r_endX = w_endX + pad_num
                    r_x = w_x - pad_num
                    r_y = w_y - pad_num
                    r_sizeX = clip_size_r
                    r_sizeY = clip_size_r

                    if (r_y <= 0):
                        pad_top = r_y * (-1)
                        r_y = 0
                    if (r_x <= 0):
                        pad_left = r_x * (-1)
                        r_x = 0
                    if (r_endY >= rows):
                        pad_bottom = r_endY - rows
                        r_endY = rows
                    if (r_endX >= cols):
                        pad_right = r_endX - cols
                        r_endX = cols

                    if (w_endY >= rows):
                        w_endY = rows
                    if (w_endX >= cols):
                        w_endX = cols

                    w_sizeX = w_endX - w_x
                    w_sizeY = w_endY - w_y
                    r_sizeX = r_endX - r_x
                    r_sizeY = r_endY - r_y

                    data = test_img_gdal.ReadAsArray(r_x, r_y, r_sizeX, r_sizeY)  # 边界小块读取的大小不为clipsize rgb



                    data = data[0:3, :, :]
                    data = np.pad(data, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                                  'edge')  # 以图片边缘像素值的方式补全至clip_size大小
                    cc, hh, ww = data.shape
                    mask = np.ones((hh, ww), dtype=np.uint8)
                    # # flag1 = np.where(data[0] == 255)
                    # # flag2 = np.where(data[1] == 255)
                    # # flag3 = np.where(data[2] == 255)
                    # mask[flag1 and flag2 and flag3] = 0
                    flag1 = np.where(data[0] == 0)
                    flag2 = np.where(data[1] == 0)
                    flag3 = np.where(data[2] == 0)
                    mask[flag1 and flag2 and flag3] = 0
                    contours, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    if np.all(data == 0):
                        out.GetRasterBand(1).WriteArray(data[0][pad_num:pad_num + w_sizeY, pad_num:pad_num + w_sizeX], w_x,
                                                        w_y)

                    elif np.all(data == 255):
                        data=np.zeros_like(data)
                        out.GetRasterBand(1).WriteArray(data[0][pad_num:pad_num + w_sizeY, pad_num:pad_num + w_sizeX], w_x, w_y)

                    else:
                        # 数据处理，为了输入网络做准备
                        data = transformN(data.transpose(1, 2, 0))
                        # data = data.astype(np.float32)
                        X = torch.Tensor(data).cuda().unsqueeze(0)

                        data_dict = {}
                        data_dict['img'] = [X]
                        # data_dict['img_metas'] = [[{'filename': '1', 'ori_filename': '1', 'ori_shape': (1280, 1280, 3),
                        #                             'img_shape': (1280, 1280, 3), 'pad_shape': (1280, 1280, 3),
                        #                             'scale_factor': [0.5, 0.5, 0.5, 0.5], 'flip': False,
                        #                             'flip_direction': 'horizontal',
                        #                             'img_norm_cfg': {'mean': [0.485, 0.456, 0.406],
                        #                                              'std': [0.229, 0.224, 0.225], 'to_rgb': True}}]]
                        data_dict['img_metas'] = [[{'filename': '1', 'ori_filename': '1', 'ori_shape': (640, 640,3),
                                                    'img_shape': (640, 640,3), 'pad_shape': (640, 640,3),
                                                    'scale_factor': [0.5, 0.5, 0.5, 0.5], 'flip': False,
                                                    'flip_direction': 'horizontal',
                                                    'img_norm_cfg': {'mean': [0.485, 0.456, 0.406],
                                                                     'std': [0.229, 0.224, 0.225], 'to_rgb': True}}]]
                        y = model(return_loss=False, rescale=True, **data_dict)

                        # fuse = F.sigmoid(y[-1]).cpu().data.numpy()[0, 0, :, :] * 255
                        fuse = y[-1]
                        fuse = mask * fuse
                        # fuse=mask * y[-1]
                        # cv2.drawContours(fuse, contours, -1, 255, 2)
                        out.GetRasterBand(1).WriteArray(fuse[pad_num:pad_num + w_sizeY, pad_num:pad_num + w_sizeX], w_x,
                                                        w_y)

        del out
        del test_img_gdal


def parse_args():
    parser = argparse.ArgumentParser('predict whole Swin transformer')
    # parser.add_argument('--config', help='train config file path',
    #                     default='../configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k.py')

    # parser.add_argument('--config', help='train config file path',
    #                     default='../configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py')
    parser.add_argument('--config', help='train config file path',
                        default=r"E:\zyk\Pyproject\UNiRepLKNet-seg\configs\swin\upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py")
    parser.add_argument('-i', '--input_path', type=str,
        default=r"E:\zyk\xj\xj_2023_8\clip", help='test')#D:\lsy\map\zhijiang D:\lsy\map\result\dushanggang\0.5m\test
    parser.add_argument('-o', '--output_path', type=str,
        default=r"D:\ai\getshp\2023\swin_1019", help='save')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')#E:\lsy\map\result\bdcn_pth\bdcn_37000.pth
    #parser.add_argument('-m', '--model', type=str, default=r"E:\zyk\Pyproject\UNiRepLKNet-seg\checkpoint\xj0906\iter_160000.pth",
    #    help='the model to test')
    # parser.add_argument('-m', '--model', type=str,
    #                     default=r"E:\zyk\Pyproject\UNiRepLKNet-seg\checkpoint\xj0830\iter_160000.pth",
    #                     help='the model to test')
    parser.add_argument('-m', '--model', type=str,
                         default=r"E:\zyk\Pyproject\UNiRepLKNet-seg\checkpoint\xj1019\iter_160000.pth",
                         help='the model to test')
    parser.add_argument('--ext', type=str, default='tif',
        help='the dir to store result')
    parser.add_argument('-k', type=int, default=1,
        help='the k-th split set of multicue')
    return parser.parse_args()


def main():
    import time
    print(time.localtime())

    args = parse_args()
    config_file=args.config
    img_root=args.input_path
    save_mask_root=args.output_path
    checkpoint_file=args.model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    tes(model, args)
    #print(img_root,save_mask_root)




if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()