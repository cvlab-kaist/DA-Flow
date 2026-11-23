import math
import random

import numpy as np
import torch
import torch.nn as nn
from .utils_ import filter2D, USMSharp
from .utils_blur import circular_lowpass_kernel, random_mixed_kernels
from .utils_noise import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from .utils_jpeg import DiffJPEG
from torch.nn import functional as F
from einops import rearrange
import av
import io


DIFFICULTY_PRESETS = {
    'easy': {
        'kernel_range': [7, 9, 11],
        'kernel_range2': [7, 9, 11],
        'blur_sigma': [0.2, 1.0],
        'blur_sigma2': [0.2, 0.6],
        'noise_range': [1, 10],
        'noise_range2': [1, 8],
        'poisson_scale_range': [0.05, 1.0],
        'poisson_scale_range2': [0.05, 0.8],
        'jpeg_range': [75, 95],
        'jpeg_range2': [75, 95],
    },
    'medium': {
        'kernel_range': [13, 15],
        'kernel_range2': [13, 15],
        'blur_sigma': [1.1, 2.0],
        'blur_sigma2': [0.7, 1.0],
        'noise_range': [11, 20],
        'noise_range2': [9, 16],
        'poisson_scale_range': [1.1, 2.0],
        'poisson_scale_range2': [0.9, 1.6],
        'jpeg_range': [55, 74],
        'jpeg_range2': [55, 74],
    },
    'hard': {
        'kernel_range': [17, 19, 21],
        'kernel_range2': [17, 19, 21],
        'blur_sigma': [2.1, 3.0],
        'blur_sigma2': [1.1, 1.5],
        'noise_range': [21, 30],
        'noise_range2': [17, 25],
        'poisson_scale_range': [2.1, 3.0],
        'poisson_scale_range2': [1.7, 2.5],
        'jpeg_range': [30, 54],
        'jpeg_range2': [30, 54],
    },
}


class ImageCompressor:
    def __init__(self):
        self.params = {
            'codec': ['libx264', 'h264', 'mpeg4'],
            'codec_prob': [1 / 3., 1 / 3., 1 / 3.],
            'bitrate': [1e4, 1e5]
        }

    def _ensure_even_dimensions(self, img):
        # Ensure width and height are even
        h, w = img.shape[:2]
        if h % 2 != 0:
            img = img[:-1, :]
        if w % 2 != 0:
            img = img[:, :-1]
        return img

    def _apply_random_compression(self, imgs):
        # imgs: Tensor [B, C, H, W]
        device = imgs.device
        # Convert PyTorch tensor to NumPy array
        # Move to CPU for processing with PyAV
        imgs_np = imgs.permute(0, 2, 3, 1).cpu().numpy()

        # Ensure width and height are even
        imgs_np = [self._ensure_even_dimensions(img) for img in imgs_np]

        codec = random.choices(self.params['codec'], self.params['codec_prob'])[0]
        bitrate = self.params['bitrate']
        bitrate = np.random.randint(bitrate[0], bitrate[1] + 1)

        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(codec, rate=1)
            stream.height = imgs_np[0].shape[0]
            stream.width = imgs_np[0].shape[1]
            stream.pix_fmt = 'yuv420p'
            stream.bit_rate = bitrate

            for img in imgs_np:
                img = (img * 255).clip(0, 255)  # Convert to [0, 255] range
                img = img.astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                try:
                    frame.pict_type = 'NONE'
                except TypeError:  # PyAV >= 14 uses an enum
                    frame.pict_type = av.video.frame.PictureType.NONE
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

        outputs = []
        with av.open(buf, 'r', 'mp4') as container:
            if container.streams.video:
                for frame in container.decode(**{'video': 0}):
                    outputs.append(frame.to_rgb().to_ndarray().astype(np.float32) / 255)  # Convert back to [0, 1] range
        
        if not outputs:
            # Fallback if decoding fails (rare)
            return imgs
            
        # Convert list to NumPy array to a single array
        outputs = np.stack(outputs, axis=0)
        # Convert NumPy array back to PyTorch tensor and move back to original device
        outputs = torch.tensor(outputs).permute(0, 3, 1, 2).to(device)

        return outputs


class Degradation(nn.Module):
    def __init__(self, scale, gt_size, difficulty=None):
        super(Degradation, self).__init__()
        ### initization JPEF class
        self.jpeger = DiffJPEG(differentiable=False) # Will be moved to device in .to(device) or forward
        self.usm_sharpener = USMSharp()
        
        ### global settings
        self.scale = scale
        self.gt_size = gt_size

        ### the first degradation hypermeters ###
        # 1. blur
        self.blur_kernel_size = 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob = 0.1
        self.blur_sigma = [0.2, 3]  # blur_x / y_sigma
        self.betag_range = [0.5, 4]
        self.betap_range = [1, 2]
        # 2. resize
        self.updown_type = ["up", "down", "keep"]
        self.mode_list = ["area", "bilinear", "bicubic"]  # flags:[3,1,2]
        self.resize_prob = [0.2, 0.7, 0.1]  # up, down, keep
        self.resize_range = [0.15, 1.5]
        # 3. noise
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.gray_noise_prob = 0.4
        # 4. jpeg
        self.jpeg_range = [30, 95]

        ### the second degradation hypermeters ###
        # 1. blur
        self.second_blur_prob = 0.8

        self.blur_kernel_size2 = 21
        self.kernel_range2 = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob2 = 0.1
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        # 2. resize
        self.updown_type2 = ["up", "down", "keep"]
        self.mode_list2 = ["area", "bilinear", "bicubic"]  # flags:[3,1,2]
        self.resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
        self.resize_range2 = [0.3, 1.2]
        # 3. noise
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.gray_noise_prob2 = 0.4
        # 4. jpeg
        self.jpeg_range2 = [30, 95]

        self.final_sinc_prob = 0.8

        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1
        # self.pulse_tensor will be moved in forward pass or if module is moved

        # video compression
        self.compressor = ImageCompressor()

        if difficulty is not None:
            self.apply_difficulty_preset(difficulty)

    def apply_difficulty_preset(self, difficulty):
        if difficulty not in DIFFICULTY_PRESETS:
            raise ValueError(f"Unsupported difficulty: {difficulty}")

        preset = DIFFICULTY_PRESETS[difficulty]
        for key, value in preset.items():
            if key == 'bitrate':
                self.compressor.params['bitrate'] = value
            else:
                setattr(self, key, value)

    @torch.no_grad()
    def forward_deg(self, gt):
        # Ensure modules are on the same device as input
        device = gt.device
        if hasattr(self, 'pulse_tensor'):
            self.pulse_tensor = self.pulse_tensor.to(device)
            
        # Manually move sub-modules if they haven't been moved (e.g. if not using .cuda() on the class)
        # But usually user calls model.to(device).
        # We ensure they are on device.
        # Note: DiffJPEG and USMSharp are nn.Modules, so they should move with .to(device).
        
        ori_h, ori_w = gt.size()[2:4]
        gt_usm = self.usm_sharpener(gt)
        gt_usm_copy = gt_usm.clone()

        # generate kernel (on device)
        kernel1 = self.generate_first_kernel().to(device)
        kernel2 = self.generate_second_kernel().to(device)
        sinc_kernel = self.generate_sinc_kernel().to(device)

        # first degradation
        # blur_1 -> resize_1 -> noise_1 -> jpeg_1 -> compression
        out = self.blur_1(gt_usm_copy, kernel1)
        out = self.resize_1(out)
        out = self.noise_1(out)
        out = self.jpeg_1(out)
        lq = self.compressor._apply_random_compression(out)

        # second degradation
        # blur_2 -> resize_2 -> noise_2 -> jpeg_2 -> compression
        out = self.blur_2(lq, kernel2)
        out = self.resize_2(out, ori_h, ori_w)
        out = self.noise_2(out)
        out = self.jpeg_2(out, ori_h, ori_w, sinc_kernel)
        lq = self.compressor._apply_random_compression(out)

        return lq, gt_usm, kernel1, kernel2, sinc_kernel

    @torch.no_grad()
    def forward(self, img_gt, uint8=False):
        # img_gt: Tensor [B, C, H, W]
        img_gt_copy = img_gt.clone()

        # degradation_piepline
        lq, gt_usm, kernel1, kernel2, sinc_kernel = self.forward_deg(img_gt_copy)
        
        # clamp and round
        lq = torch.clamp((lq * 255.0).round(), 0, 255) / 255.
        
        return lq, gt_usm


    def blur_1(self, img, kernel1):
        img = filter2D(img, kernel1)
        return img

    def blur_2(self, img, kernel2):
        if np.random.uniform() < self.second_blur_prob:
            img = filter2D(img, kernel2)
        return img

    def resize_1(self, img):
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        img = F.interpolate(img, scale_factor=scale, mode=mode)

        return img

    def resize_2(self, img, ori_h, ori_w):
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        # Calculate size to match target scale
        img = F.interpolate(
            img, size=(int(ori_h / self.scale * scale), int(ori_w / scale * scale)), mode=mode)

        return img

    def noise_1(self, img):
        gray_noise_prob = self.gray_noise_prob
        if np.random.uniform() < self.gaussian_noise_prob:
            img = random_add_gaussian_noise_pt(img, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            img = random_add_poisson_noise_pt(
                img,
                scale_range=self.poisson_scale_range,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        return img

    def noise_2(self, img):
        gray_noise_prob = self.gray_noise_prob2
        if np.random.uniform() < self.gaussian_noise_prob2:
            img = random_add_gaussian_noise_pt(
                img, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            img = random_add_poisson_noise_pt(
                img,
                scale_range=self.poisson_scale_range2,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        return img

    def jpeg_1(self, img):
        jpeg_p = img.new_zeros(img.size(0)).uniform_(*self.jpeg_range)
        img = torch.clamp(img, 0, 1)
        img = self.jpeger(img, quality=jpeg_p)
        return img

    def jpeg_2(self, out, ori_h, ori_w, sinc_kernel):
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            out = filter2D(out, sinc_kernel)

        return out

    def generate_first_kernel(self):
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return torch.FloatTensor(kernel)

    def generate_second_kernel(self):
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        return torch.FloatTensor(kernel2)

    def generate_sinc_kernel(self):
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        return sinc_kernel

    # Helpers removed or kept minimal as they are not used in main flow anymore
    
# init degradation process
# Initialize on CPU first
_DEGRADATION_CACHE = {}


def get_degradation(difficulty=None):
    cache_key = difficulty or 'default'
    if cache_key not in _DEGRADATION_CACHE:
        _DEGRADATION_CACHE[cache_key] = Degradation(scale=4, gt_size=(480, 720), difficulty=difficulty)
    return _DEGRADATION_CACHE[cache_key]


def degradation_process(video_array, difficulty=None):
    # video_array: [B, C, T, H, W] on Device
    _, _, t, _, _ = video_array.shape
    device = video_array.device
    degradation = get_degradation(difficulty)
    
    # Move degradation module to the same device as input video
    degradation.to(device)

    # preprocess video
    video_array = video_array.to(torch.float32)
    video_array = (video_array + 1) * 0.5 # [-1, 1] -> [0, 1]
    
    # Rearrange to process frames as batch
    video_array = rearrange(video_array, "B C T H W -> (B T) C H W")
    
    # Clamp just in case
    video_array = torch.clamp(video_array, 0, 1)

    # Degrade in chunks of frames to bound GPU memory
    chunk_size = 8
    lq_list = []
    num_frames = video_array.shape[0]
    for i in range(0, num_frames, chunk_size):
        lq, _ = degradation(video_array[i:i + chunk_size])
        lq_list.append(lq)
    lq = torch.cat(lq_list, dim=0).clip(0, 1) * 2 - 1
    lq = rearrange(lq, "(B T) C H W -> B C T H W", T=t).to(torch.float32)
    return lq, video_array
