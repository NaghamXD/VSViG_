# extract patches based on keypoints

import math, cv2
import numpy as np

def norm(x):
    # Added epsilon to avoid division by zero if max == min (e.g. uniform patch)
    denominator = np.max(x) - np.min(x)
    if denominator == 0:
        return np.zeros_like(x)
    
    y = ((x - np.min(x)) / denominator) * 255
    return y

def gen_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    kernel = kernel / np.sum(kernel)
    kernel = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))
    return kernel

def extract_patches(img, kpts, kernel_size = 128, kernel_sigma=0.3, scale=1/4):
    img_shape = img.shape # 1080 x 1920 x 3 video resolution
    pad_img = np.zeros((img_shape[0]+kernel_size*2, img_shape[1]+kernel_size*2, 3))
    pad_img[kernel_size:-kernel_size, kernel_size:-kernel_size, :] = img
    kernel = gen_kernel(kernel_size,kernel_size*kernel_sigma)
    kernel = np.expand_dims(kernel,2).repeat(3,axis=2)
   #kpts = np.delete(kpts, [[1],[-3],[-4]], axis=0) # 
    patches = np.zeros((15,math.ceil(scale*kernel_size),math.ceil(scale*kernel_size),3))
    for idx in range(15):
        # Coordinates in padded image
        # Note: kpts are in original image coordinates
        y_center = int(kpts[idx,1] + kernel_size)
        x_center = int(kpts[idx,0] + kernel_size)
        
        # Crop region: 0.5*kernel_size to 1.5*kernel_size around keypoint
        # This effectively centers the crop on the keypoint if kpts were shifted
        # The logic: center is at kpts + kernel_size
        # We want range [center - 0.5*kernel_size, center + 0.5*kernel_size]
        # Which is [kpts + 0.5*kernel_size, kpts + 1.5*kernel_size]
        
        y1 = int(kpts[idx,1] + 0.5*kernel_size)
        y2 = int(kpts[idx,1] + 1.5*kernel_size)
        x1 = int(kpts[idx,0] + 0.5*kernel_size)
        x2 = int(kpts[idx,0] + 1.5*kernel_size)
        
        crop = pad_img[y1:y2, x1:x2, :]
        
        # Safety for slight rounding errors in size
        if crop.shape != kernel.shape:
             crop = cv2.resize(crop, (kernel_size, kernel_size))
             
        tmp = norm(crop * kernel)
        
        # Handle potential NaNs from norm just in case
        if np.isnan(tmp).any():
            tmp = np.nan_to_num(tmp, nan=0.0)
            
        tmp = cv2.resize(tmp, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        patches[idx,:,:,:] = tmp
    return patches