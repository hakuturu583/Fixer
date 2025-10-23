# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import torchvision.transforms as transforms
from PIL import Image
import torch
import os
from glob import glob
from tqdm import tqdm
import transformer_engine as te # 1.11.0+4df8488
import time
import torchvision.transforms.functional as F

def preprocess_nre_fixer(input_image, res):
    if res == 1024:
        input_image = input_image.resize((1024, 576), Image.BILINEAR)
    elif res == 832:
        input_image = input_image.resize((832, 448), Image.BILINEAR)
    elif res == 704:
        input_image = input_image.resize((704, 384), Image.BILINEAR)
    elif res == 512:
        input_image = input_image.resize((512, 288), Image.BILINEAR) 
    else:
        raise Exception('RESOLUTION NOT SUPPORTED')
        
    input_tensor = F.to_tensor(input_image).unsqueeze(0).cuda()
    input_tensor = input_tensor * 2 - 1  
    return input_tensor

def main():
    parser = argparse.ArgumentParser(description='Run inference on an image.')
    parser.add_argument('--input', type=str, help='Path to input directory or image', 
                        default='input/')
    parser.add_argument('--output', type=str, help='Output directory', 
                        default='output/')
    parser.add_argument('--resolution', type=int, help='Resolution of the input image', 
                        default=1024)
    parser.add_argument('--model', type=str, help='Model path', required=True)

    
    args = parser.parse_args()

    assert te, "WARNING: Make sure to import transformer_engine as te"
    model = torch.jit.load(args.model)
    model = model.eval().cuda().to(torch.bfloat16)        
        
        
    ### Run inference
    all_img_paths = glob(args.input + '/*.png') + glob(args.input + '/*.jpeg')
    all_img_paths.append(args.input)
    all_img_paths = [tmp for tmp in all_img_paths if 'png' in tmp or 'jpeg' in tmp]
    
    os.makedirs(args.output, exist_ok=True)
    
    inference_times = []
    
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(all_img_paths, desc="Processing images")):
            input_image = Image.open(img_path).convert('RGB')
            
            orginal_shape = input_image.size

            input_tensor = preprocess_nre_fixer(input_image, args.resolution)
            input_tensor = input_tensor.to(torch.bfloat16)

            if i == 0:
                print(f"Input dtype: {input_tensor.dtype}, shape: {input_tensor.shape}\n")

            start_time = time.time()
            output_image = model(input_tensor).float()
            elapsed = time.time() - start_time
            inference_times.append(elapsed)

            output_image = output_image[0].cpu() * 0.5 + 0.5
            output_image = torch.clamp(output_image, 0.0, 1.0)
            
            output_pil = transforms.ToPILImage()(output_image)
            output_pil = output_pil.resize(orginal_shape)

            # Fix: Use os.path.join to properly handle directory separator
            output_filename = os.path.basename(img_path)
            output_path = os.path.join(args.output, output_filename)
            output_pil.save(output_path)
            
            # Show timing for every image using tqdm.write (doesn't break progress bar)
            tqdm.write(f"  {output_filename}: {elapsed:.4f}s")
        
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    print(f'\n✓ Successfully processed {len(all_img_paths)} images')
    print(f'Average inference time: {avg_time:.4f}s per image')

if __name__ == '__main__':
    main()
