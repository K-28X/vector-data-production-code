import numpy as np
import yaml
import argparse
import glob
from osgeo import gdal
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from denoising_diffusion_pytorch.utils import *
import torchvision as tv
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
# from denoising_diffusion_pytorch.transmodel import TransModel
from denoising_diffusion_pytorch.uncond_unet import Unet
from denoising_diffusion_pytorch.data import *
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from fvcore.common.config import CfgNode
from scipy import integrate
import cv2
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="training vae configure")
    # parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    parser.add_argument("--cfg", help="experiment configure file name", type=str)
    # parser.add_argument("")
    args = parser.parse_args()
    args.cfg = "./configs/BSDS_sample.yaml"
    args.cfg = load_conf(args.cfg)

    # args.cfg['data']['img_folder'] = r"E:\BaiduNetdiskDownload\DiffusionEdge-main\data\test"
    # args.cfg['data']['output_path'] = r"E:\BaiduNetdiskDownload\DiffusionEdge-main\results"
    print(args.cfg)
    return args


def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf

# Colors for all 20 parts
part_colors = [[0, 0, 0], [255, 85, 0],  [255, 170, 0],
               [255, 0, 85], [255, 0, 170],
               [0, 255, 0], [85, 255, 0], [170, 255, 0],
               [0, 255, 85], [0, 255, 170],
               [0, 0, 255], [85, 0, 255], [170, 0, 255],
               [0, 85, 255], [0, 170, 255],
               [255, 255, 0], [255, 255, 85], [255, 255, 170],
               [255, 0, 255], [255, 85, 255], [255, 170, 255],
               [0, 255, 255], [85, 255, 255], [170, 255, 255]]


def main(args):
    cfg = CfgNode(args.cfg)
    torch.manual_seed(42)
    np.random.seed(42)
    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.ckpt_path,
    )

    if model_cfg.model_name == 'cond_unet':
        from denoising_diffusion_pytorch.mask_cond_unet import Unet
        unet_cfg = model_cfg.unet
        unet = Unet(dim=unet_cfg.dim,
                    channels=unet_cfg.channels,
                    dim_mults=unet_cfg.dim_mults,
                    learned_variance=unet_cfg.get('learned_variance', False),
                    out_mul=unet_cfg.out_mul,
                    cond_in_dim=unet_cfg.cond_in_dim,
                    cond_dim=unet_cfg.cond_dim,
                    cond_dim_mults=unet_cfg.cond_dim_mults,
                    window_sizes1=unet_cfg.window_sizes1,
                    window_sizes2=unet_cfg.window_sizes2,
                    fourier_scale=unet_cfg.fourier_scale,
                    cfg=unet_cfg,
                    )
    else:
        raise NotImplementedError
    if model_cfg.model_type == 'const_sde':
        from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
    else:
        raise NotImplementedError(f'{model_cfg.model_type} is not surportted !')
    ldm = LatentDiffusion(
        model=unet,
        auto_encoder=first_stage_model,
        train_sample=model_cfg.train_sample,
        image_size=model_cfg.image_size,
        timesteps=model_cfg.timesteps,
        sampling_timesteps=model_cfg.sampling_timesteps,
        loss_type=model_cfg.loss_type,
        objective=model_cfg.objective,
        scale_factor=model_cfg.scale_factor,
        scale_by_std=model_cfg.scale_by_std,
        scale_by_softsign=model_cfg.scale_by_softsign,
        default_scale=model_cfg.get('default_scale', False),
        input_keys=model_cfg.input_keys,
        ckpt_path=model_cfg.ckpt_path,
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
        perceptual_weight=model_cfg.perceptual_weight,
        use_l1=model_cfg.get('use_l1', True),
        cfg=model_cfg,
    )
    # ldm.init_from_ckpt(cfg.sampler.ckpt_path, use_ema=cfg.sampler.get('use_ema', True))

    data_cfg = cfg.data
    # print(data_cfg)
    # if data_cfg['name'] == 'edge':
    #     dataset = EdgeDatasetTest(
    #         data_root=data_cfg.img_folder,
    #         image_size=model_cfg.image_size,
    #     )
    #     # dataset = torch.utils.data.ConcatDataset([dataset] * 5)
    # else:
    #     raise NotImplementedError

    # print(dataset)

    # dl = DataLoader(dataset, batch_size=cfg.sampler.batch_size, shuffle=False, pin_memory=True,
    #                 num_workers=data_cfg.get('num_workers', 2))

    # print(dl)
    sampler_cfg = cfg.sampler
    sampler = Sampler(
        ldm, data_cfg, batch_size=sampler_cfg.batch_size,
        sample_num=sampler_cfg.sample_num,
        results_folder=sampler_cfg.save_folder,cfg=cfg,
    )
    sampler.sample()
    if data_cfg.name == 'cityscapes' or data_cfg.name == 'sr' or data_cfg.name == 'edge':
        exit()
    else:
        assert len(os.listdir(sampler_cfg.target_path)) > 0, "{} have no image !".format(sampler_cfg.target_path)
        sampler.cal_fid(target_path=sampler_cfg.target_path)
    pass


class Sampler(object):
    def __init__(
            self,
            model,
            data_loader,
            sample_num=1000,
            batch_size=16,
            results_folder='./results',
            rk45=False,
            cfg={},
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='no',
            kwargs_handlers=[ddp_handler],
        )
        self.model = model
        self.sample_num = sample_num
        self.rk45 = rk45

        self.batch_size = batch_size
        self.batch_num = math.ceil(sample_num // batch_size)

        self.image_size = model.image_size
        self.cfg = cfg

        # dataset and dataloader

        # self.ds = Dataset(folder, mask_folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        self. data_loader = data_loader
        self.results_folder = Path(results_folder)
        # self.results_folder_cond = Path(results_folder+'_cond')
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)
            # self.results_folder_cond.mkdir(exist_ok=True, parents=True)

        self.model = self.accelerator.prepare(self.model)
        data = torch.load(cfg.sampler.ckpt_path, map_location=lambda storage, loc: storage)

        model = self.accelerator.unwrap_model(self.model)
        if cfg.sampler.use_ema:
            sd = data['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]  # remove ema_model.
                    new_sd[new_k] = sd[k]
            sd = new_sd
            model.load_state_dict(sd)
        else:
            # print(data['model'])
            model.load_state_dict(data['model'])
        if 'scale_factor' in data['model']:
            model.scale_factor = data['model']['scale_factor']



    def sample(self):
        accelerator = self.accelerator
        device = accelerator.device
        batch_num = self.batch_num
        print(self.data_loader)
        input_path = self.data_loader.img_folder
        input_paths = glob.glob(os.path.join(input_path, '*.{}'.format('tif')))
        print(input_paths)
        print(len(input_paths))

        if not os.path.exists(self.data_loader.output_path):
            os.makedirs(self.data_loader.output_path)

        for input_path in tqdm(input_paths):
            test_path = input_path
            test_img_gdal = gdal.Open(test_path)
            model = self.model.cuda()
            model.eval()
            clip_size_r = 2048  # 128#256#512#1024 512+256
            # pad_num = clip_size_r // 4
            pad_num = 80
            clip_size_w = clip_size_r - pad_num  # 256
            cols = test_img_gdal.RasterXSize
            rows = test_img_gdal.RasterYSize
            rows_iters = rows // clip_size_w + 1
            cols_iters = cols // clip_size_w + 1
            print(rows, cols)
            print(rows_iters, cols_iters)
            geo = test_img_gdal.GetGeoTransform()
            proj = test_img_gdal.GetProjection()
            outpath = os.path.join(self.data_loader.output_path, os.path.basename(test_path))
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

                        data = test_img_gdal.ReadAsArray(r_x, r_y, r_sizeX, r_sizeY)  # 边界小块读取的大小不为clipsize

                        data = data[0:3, :, :]
                        data = np.pad(data, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                                      'edge')  # 以图片边缘像素值的方式补全至clip_size大小
                        if np.all(data == 0):
                            out.GetRasterBand(1).WriteArray(
                                data[0][pad_num:pad_num + w_sizeY, pad_num:pad_num + w_sizeX], w_x,
                                w_y)

                        elif np.all(data == 255):
                            data = np.zeros_like(data)
                            out.GetRasterBand(1).WriteArray(
                                data[0][pad_num:pad_num + w_sizeY, pad_num:pad_num + w_sizeX], w_x,
                                w_y)
                        else:
                            cc, hh, ww = data.shape
                            mask = np.ones((hh, ww), dtype=np.uint8)
                            # 找到三个通道中像素值为 255 的位置
                            # flag1 = np.where(data[0] == 255)
                            # flag2 = np.where(data[1] == 255)
                            # flag3 = np.where(data[2] == 255)
                            # mask[flag1 and flag2 and flag3] = 0
                            flag1 = np.where(data[0] == 0)
                            flag2 = np.where(data[1] == 0)
                            flag3 = np.where(data[2] == 0)
                            mask[flag1 and flag2 and flag3] = 0
                            contours, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            # 数据处理，为了输入网络做准备

                            # data = transformN(data.transpose(1,2,0))
                            # X = torch.Tensor(data).cuda().unsqueeze(0)
                            X = data
                            # print(X)
                            # print(f"some_method called with X of shape {X.shape}")
                            m = transformN(X.transpose((1, 2, 0)))
                            # print(m)
                            # print(f"some_method called with m of shape {m.shape}")
                            data_dict = {}
                            d = {}
                            # d['img'] = Image.fromarray(data.transpose((1, 2, 0)))
                            # a = normal(d)
                            # a = transform(d)
                            device = torch.device('cuda:0')
                            img = torch.Tensor(m).cuda().unsqueeze(0)

                            cond = img.to(device)
                            # print(cond)
                            # print(f"some_method called with cond of shape {cond.shape}")
                            # raw_w = # default batch size = 1
                            # raw_h = 320
                            mask2 =  None

                            batch_pred = self.slide_sample(cond,
                                                           crop_size=self.cfg.sampler.get('crop_size', [320, 320]),
                                                           stride=self.cfg.sampler.stride, mask=mask2)

                            seg = batch_pred.detach().cpu().numpy()
                            # print(seg)
                            # print(f"some_method called with seg of shape {seg.shape}")
                            # fl = 0.20
                            # seg[seg > fl] = 1
                            # seg[seg <= fl] = 0
                            # seg = 255 - seg * 255
                            # fuse = torch.sigmoid(torch.tensor(y[-1])).cpu().data.numpy()[0, :, :] * 255
                            seg = seg.squeeze()
                            seg = seg * 255.0

                            fuse = mask * seg
                            # cv2.drawContours(fuse, contours, -1, 255, 2)
                            out.GetRasterBand(1).WriteArray(fuse[pad_num:pad_num + w_sizeY, pad_num:pad_num + w_sizeX],
                                                            w_x,
                                                            w_y)

            del out
            del test_img_gdal

        accelerator.print('sampling complete')

    # ----------------------------------waiting revision------------------------------------



    def slide_sample(self, inputs, crop_size, stride, mask=None):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = 1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        aux_out1 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        # aux_out2 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]

                if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    crop_seg_logit = self.model.module.sample(batch_size=1, cond=crop_img, mask=mask)
                    e1 = e2 = None
                    aux_out = None
                elif isinstance(self.model, nn.Module):
                    crop_seg_logit = self.model.sample(batch_size=1, cond=crop_img, mask=mask)
                    e1 = e2 = None
                    aux_out = None
                else:
                    raise NotImplementedError
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                if aux_out is not None:
                    aux_out1 += F.pad(aux_out,
                                   (int(x1), int(aux_out1.shape[3] - x2), int(y1),
                                    int(aux_out1.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        # torch.save(count_mat, '/home/yyf/Workspace/edge_detection/codes/Mask-Conditioned-Latent-Space-Diffusion/checkpoints/count.pt')
        seg_logits = preds / count_mat
        aux_out1 = aux_out1 / count_mat
        # aux_out2 = aux_out2 / count_mat
        if aux_out is not None:
            return seg_logits, aux_out1
        return seg_logits


    def whole_sample(self, inputs, raw_size, mask=None):

        inputs = F.interpolate(inputs, size=(416, 416), mode='bilinear', align_corners=True)

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            seg_logits = self.model.module.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        elif isinstance(self.model, nn.Module):
            seg_logits = self.model.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        seg_logits = F.interpolate(seg_logits, size=raw_size, mode='bilinear', align_corners=True)
        return seg_logits


    def cal_fid(self, target_path):
        command = 'fidelity -g 0 -f -i -b {} --input1 {} --input2 {}'\
            .format(self.batch_size, str(self.results_folder), target_path)
        os.system(command)

    def rk45_sample(self, batch_size):
        with torch.no_grad():
            # Initial sample
            # z = torch.randn(batch_size, 3, *(self.image_size))
            shape = (batch_size, 3, *(self.image_size))
            ode_sampler = get_ode_sampler(method='RK45')
            x, nfe = ode_sampler(model=self.model, shape=shape)
            x = unnormalize_to_zero_to_one(x)
            x.clamp_(0., 1.)
            return x, nfe

def get_ode_sampler(rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  # def denoise_update_fn(model, x):
  #   score_fn = get_score_fn(sde, model, train=False, continuous=True)
  #   # Reverse diffusion predictor for denoising
  #   predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
  #   vec_eps = torch.ones(x.shape[0], device=x.device) * eps
  #   _, x = predictor_obj.update_fn(x, vec_eps)
  #   return x

  def drift_fn(model, x, t, model_type='const'):
    """Get the drift function of the reverse-time SDE."""
    # score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # rsde = sde.reverse(score_fn, probability_flow=True)
    pred = model(x, t)
    if model_type == 'const':
        drift = pred
    elif model_type == 'linear':
        K, C = pred.chunk(2, dim=1)
        drift = K * t + C
    return drift

  def ode_sampler(model, shape):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = torch.randn(*shape)
      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        # vec_t = torch.ones(shape[0], device=x.device) * t
        vec_t = torch.ones(shape[0], device=x.device) * t * 1000
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (1, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      # if denoise:
      #   x = denoise_update_fn(model, x)

      # x = inverse_scaler(x)
      return x, nfe

  return ode_sampler

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))
# 这段代码的目的是将输入的图像进行一系列预处理操作，包括像素值的缩放、颜色通道的标准化、尺寸调整等，以便使图像数据适合用于某个特定的深度学习模型进行训练或推断。
def transformN(img):
        # 将图像的像素值从整数范围 [0, 255] 缩放到浮点数范围 [0.0, 1.0]，以便在后续处理中更方便地进行数值计算。
        img = img / 255.0
        # mean_bgr 和 std_bgr：这两个数组分别代表了图像在每个颜色通道上的平均值和标准差。

        # 将图像数据转换为浮点数类型，以便进行后续的数值计算。
        img = np.array(img, dtype=np.float32)
        # 对图像每个通道分别减去相应的平均值，这样做可以将图像数据的平均值近似调整为零。
        img = img * 2 - 1
        # 将图像的维度顺序从(Height, Width, Channels)
        # 调整为(Channels, Height, Width)，以便与深度学习模型的输入格式相匹配。
        img = img.transpose((2, 0, 1))
        # 将 NumPy 数组转换为 PyTorch 的张量，并确保数据类型为浮点数类型。
        img = torch.from_numpy(img.copy()).float()
        return img

if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass