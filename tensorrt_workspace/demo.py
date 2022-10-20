import os
import sys
import argparse

import numpy as np
import tensorrt as trt
from common import *
import torch
from fastflownet import centralize

import pycuda.driver as cuda
import pycuda.autoinit

import ctypes
ctypes.CDLL(open(os.path.join(os.path.dirname(__file__),'tensorrt_plugin_path')).read().rstrip('\n'))

import glob

def center_crop(img, set_size):

    h, w, c = img.shape

    if set_size > min(h, w):
        return img

    crop_width = set_size
    crop_height = set_size

    mid_x, mid_y = w//2, h//2
    offset_x, offset_y = crop_width//2, crop_height//2

    crop_img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
    return crop_img

def prepare_batch(path_list, H=512, W=512, max_batch=8):
    # to change this to torch dataloader, move reading file part to loader 
    img1_list = []
    img2_list = []
    basename_list = []
    batch_list = []
    for img1_path, img2_path in zip(path_list[:-1], path_list[1:]):
        img1_tensor = torch.from_numpy(center_crop(cv2.imread(img1_path), H)).float().permute(2,0,1)[None] / 255.
        img2_tensor = torch.from_numpy(center_crop(cv2.imread(img2_path), H)).float().permute(2,0,1)[None] / 255.
        img1_list.append(img1_tensor)
        img2_list.append(img2_tensor)
        basename_list.append(os.path.basename(img1_path))
        
        if len(img1_list) == max_batch:
            img1_tensor = torch.cat(img1_list, 0)
            img2_tensor = torch.cat(img2_list, 0)
            img1, img2, _ = centralize(img1_tensor, img2_tensor)
            batch_list.append((torch.cat([img1, img2], 1), basename_list))
            img1_list = []
            img2_list = []
            basename_list = []

    return batch_list

if __name__ == '__main__':
    dataset_path = "/data/datasets/homeplus_fisheye_2_downgrade_fisheye/*.png"
    out_path = "/data/output/CAOD"
    logger = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(logger, '')
    script_root = os.path.dirname(__file__)
    img_path_list = glob.glob(dataset_path)
    max_batch = 32
    with open('./engine_fp16', "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        inputs,outputs,bindings,stream = allocate_buffers(engine,True,2)

        import cv2
        import time
        from flow_vis import flow_to_color
        div_flow = 20.0
        div_size = 64

        for binding in engine:
            print('-------------------')
            print(engine.get_binding_shape(binding))
            print(engine.get_binding_name(engine.get_binding_index(binding)))

        img_path_list = sorted(img_path_list)
        #H = (540 // 64 + 1) * 64
        #W = (960 // 64 + 1) * 64
        H = 512
        W = 512
        batch_list = prepare_batch(img_path_list[:512], H=H, W=W, max_batch=max_batch)
        
        with engine.create_execution_context() as context:

            cost = 0
            for input_t, basename_list in batch_list:
                input_t = input_t.float().numpy()
                inputs[0].host = input_t
                tic = time.time()
                trt_outputs = do_inference_v2(context,bindings=bindings,inputs=inputs,outputs=outputs,stream=stream)
                toc = time.time()
                output = trt_outputs[0].reshape(engine.get_binding_shape(1))
                
                for i, basename in enumerate(basename_list):
                    flow = div_flow * output[i]
                    flow = np.transpose(flow,[1,2,0])
                    flow_color = flow_to_color(flow,convert_to_bgr=True)
                    cv2.imwrite(os.path.join(out_path, basename), flow_color)

                cost += toc - tic

            print('fps: ',1/((cost)/(len(batch_list) * max_batch)))
