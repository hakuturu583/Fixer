# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo_nocond_cosmos_base_2x_smallimg_skip import Pix2Pix_Turbo

from glob import glob
from tqdm import tqdm
import imageio
from utils.training_utils import build_transform

from natsort import natsorted

### color matching
import cv2
import numpy as np
from skimage import exposure
from PIL import Image

def histogram_matching(source, target):
    """
    Adjust the pixel values of a source image so that its histogram matches that of a target image.

    Parameters:
    - source: NumPy array of the source image.
    - target: NumPy array of the target image.

    Returns:
    - Matched image as a NumPy array.
    """
    matched = np.zeros_like(source)
    for channel in range(source.shape[2]):
        src_channel = source[:, :, channel]
        tgt_channel = target[:, :, channel]

        matched[:, :, channel] = exposure.match_histograms(src_channel, tgt_channel)
    return matched

def color_transfer(source, target):
    """
    Transfers the color distribution from the target image to the source image
    using the method of Reinhard et al., 2001.

    Parameters:
    - source: NumPy array of the source image.
    - target: NumPy array of the target image.

    Returns:
    - Color transferred image as a NumPy array.
    """
    # Convert images from BGR to L*a*b* color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(float)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(float)

    # Compute the mean and standard deviation of each channel
    src_mean, src_std = cv2.meanStdDev(source_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(target_lab)

    # Reshape mean and std to match image dimensions
    src_mean = src_mean.reshape((1, 1, 3))
    src_std = src_std.reshape((1, 1, 3))
    tgt_mean = tgt_mean.reshape((1, 1, 3))
    tgt_std = tgt_std.reshape((1, 1, 3))

    # Subtract the mean from the source image
    lab = source_lab - src_mean

    # Scale by the standard deviation ratio
    lab = (lab * (tgt_std / src_std))

    # Add the target mean
    lab += tgt_mean

    # Clip the values to valid range
    lab = np.clip(lab, 0, 255)

    # Convert back to BGR color space
    transferred = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return transferred

    
def match_color(source_path, target_path):
    size = Image.open(source_path).size
    source = np.array(Image.open(source_path).resize(size))[:, :, ::-1]
    target = np.array(Image.open(target_path).resize(size))[:, :, ::-1]


    # Check if images are loaded
    if source is None or target is None:
        print("Error: Could not load images.")
        exit()

    # Resize images if needed (optional)
    # source = cv2.resize(source, (target.shape[1], target.shape[0]))

    # Apply histogram matching
    matched_image = histogram_matching(source, target)
    color_transferred_image = color_transfer(source, target)
    cv2.imwrite(source_path, color_transferred_image)
    print('saving matched image to', source_path)


def save_folder2video(save_dir, remove_key = 0):
    if remove_key==0:
        video_file = os.path.join(save_dir,  "video.mp4")
    else:
        video_file = os.path.join(save_dir,  "video_removekey_" + str(remove_key) + ".mp4")
    im_files = natsorted(glob(os.path.join(save_dir, "*.png")) + glob(os.path.join(save_dir, "*.jpg")) + glob(os.path.join(save_dir, "*.jpeg")))
    print('im_files', im_files)
    # Loads all images at once - can run out of memory if too many frames
    ims: list[np.ndarray] = [imageio.v2.imread(f) for f in im_files]

    # chop image if dimension not divisible by 2 to fix the ffmpeg error
    for i in range(len(ims)):
        if ims[i].shape[0] % 2 == 1:
            ims[i] = ims[i][:-1, :, :]
        if ims[i].shape[1] % 2 == 1:
            ims[i] = ims[i][:, :-1, :]

    # Results in an Array is not the same as ArrayLike annotation error, so we suppress it
    if remove_key!=0:
        ims = [ims[i] for i in range(len(ims)) if i % remove_key != 0]
        
    
    imageio.v2.mimwrite(video_file, ims, fps=30, macro_block_size=1)  # type: ignore

import time
from torch.cuda.amp import autocast


def inference_wrapper(c_t):
       
    orig_dtype = c_t.dtype         # Store the original dtype
    c_t = c_t.to(torch.bfloat16)      # Cast to bf16 for computation
    output_image = model(c_t) # [1, 2, 3, 576, 1024] 
    
            
    return output_image.to(orig_dtype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input directory')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--timestep', type=int, default=400, help='Diffusion timestep')
    parser.add_argument('--max_frames', type=int, default=3000000, help='Diffusion timestep')
    parser.add_argument('--skip_frames', type=int, default=1, help='Diffusion timestep')
    parser.add_argument('--match_color', action='store_true')
    parser.add_argument('--use_sched', action='store_true')
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--vae_skip_connection', action='store_true')
    parser.add_argument('--save_video', action='store_true')
      

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # initialize the model
    model = Pix2Pix_Turbo(pretrained_path=args.model, 
                          timestep=args.timestep, 
                          #add_noise=False,
                          train_full_unet=True, 
                          freeze_vae=False, 
                          vae_skip_connection=args.vae_skip_connection)

    model.set_eval()
    model = model.cuda().to(torch.bfloat16)
    
    model = torch.compile(model)

    # translate the image
    if args.save_video:
        save_folder2video(args.input)
    
    all_img_paths = glob(args.input + '/*.png') + glob(args.input + '/*.jpg') + glob(args.input + '/*.jpeg')
    all_img_paths.sort()
    
    print(f"\nProcessing {len(all_img_paths[:args.max_frames][::args.skip_frames])} images...")
    print(f"Resolution: {args.resolution}, Timestep: {args.timestep}\n")
    
    inference_times = []
    
    for i, img_path in enumerate(tqdm(all_img_paths[:args.max_frames][::args.skip_frames], desc="Processing images")):
        
        assert args.resolution in [960, 1024, 704, 1360, 512]
        if args.resolution == 960:
            size = (960, 544)
        elif args.resolution == 1360:
            size = (1360, 768)
        elif args.resolution == 704:
            size = (704, 384)
        elif args.resolution == 512:
            size = (512, 288)
        else:
            size = (1024, 576)
        
        input_image = Image.open(img_path).convert('RGB')
        orginal_shape = input_image.size
        input_image = input_image.resize(size, Image.BILINEAR)
        
        
        bname = os.path.basename(img_path)
      
        def get_ct(img):
            c_t = transforms.ToTensor()(img)    
            c_t = transforms.Normalize([0.5], [0.5])(c_t).unsqueeze(0).cuda()
            return c_t
        
        #with torch.no_grad():
        with torch.no_grad(), autocast(dtype=torch.bfloat16):

            c_t = get_ct(input_image) 
            c_t = c_t.to(torch.bfloat16)

            start_time = time.time()
            output_image = inference_wrapper(c_t)
            elapsed = time.time() - start_time
            inference_times.append(elapsed)

            output_image = output_image.float()
            output_image = output_image[0].cpu() * 0.5 + 0.5
            output_image = torch.clamp(output_image, 0.0, 1.0)
            
            output_pil = transforms.ToPILImage()(output_image)
            output_pil = output_pil.resize(orginal_shape)

        # save the output image
        sv_path = os.path.join(args.output, bname)
        output_pil.save(sv_path)
        
        # Show timing for each image
        tqdm.write(f"  {bname}: {elapsed:.4f}s")
        
        if args.match_color:
            match_color(sv_path, img_path)
    
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    print(f'\n✓ Successfully processed {len(all_img_paths[:args.max_frames][::args.skip_frames])} images')
    print(f'Average inference time: {avg_time:.4f}s per image')
    
    if args.save_video:
        print("Creating video...")
        save_folder2video(args.output)
