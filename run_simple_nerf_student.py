'''
READ THIS

you should run thhis python code from Command Prompt.

To install pytorch-cpu: In anaconda run the following :`conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cpuonly -c pytorch`

if you have gpu you can run also your code in GPU:
you can isntall pytorch-gpu (based on cuda version)
For example 

# CUDA 11.6
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# CUDA 11.7
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

For more information: 
https://pytorch.org/get-started/previous-versions/


There are bunch of video on yourtuve that show how to install pytorch.

'''


import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_blender import load_blender_data



def create_pointcloud(N, min_val, max_val):
    '''
    N - number of points along the axis
    min_vaL and max_val: points are sampled between these two points
    '''

    t = np.linspace(min_val, max_val, N)
    t_index = np.arange(N)
    query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
    query_indices = np.stack(np.meshgrid(t_index, t_index, t_index), -1).astype(np.int16)

    flat = query_pts.reshape([-1,3])
    flat_indices = query_indices.reshape([-1,3])

    print("--" * 70)
    print(f"[INFO.create_pointcloud]:\t Point cloud ... Values = {flat.shape}, t= [{min_val},{max_val},{N}], indices = {flat_indices.shape}")
    print("--" * 70)

    return torch.tensor(flat), t, flat_indices

def raw2outputs(raw, z_vals, rays_d):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: Predicted rgb and density of size (num_rays, num_samples along ray, 4).
        z_vals: Integration time of size (num_rays, num_samples along ray).
        rays_d:Direction of each ray of size (num_rays, 3)
    Returns:
        rgb_map: Estimated RGB color of a ray of size (num_rays, 3.)
    """

 
    '''
    TODO
    Implement volume rendering
    '''
    '''
    by weights we mean the all coefficients the multiplies the color \hat C(r) = \sum_{n=1}^{N} T_n (1-\exp(-\sigma_n\delta_n))c_n
    weights_n = T_n (1-\exp(-\sigma_n\delta_n))
    '''
    sigma_n = raw[:,3]
    rgb = raw[:,:3]
    # T_n = 1.-tf.exp(-sigma_a * dists) 
    weights = raw[:,4] * (1.0 - torch.exp(-sigma_n * z_vals)) # size of (num_rays, num_samples along ray). 
    rgb_map = None
    
    acc_map = torch.sum(weights, -1)

    if True: # this MUST BE ALWAYS TRUE in your case
        '''
        for the pixel in the background the density values is around zero.
        if the density values are all zero then the rgb_map and acc_map will be zero and the output will be 1.
        on the other hand if the acc_map is 1 (the presence of the object) the last term will be zero!
        '''
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map

def find_nn(pts, ptscloud):
    """
    pts: points along the ray of size (M,3)
    ptscloud: points in the pointcloud (KX3), where K=200X200X200


    :returns nn_index: the nearest index for every point in pts 
    """
    nn_index = []
    M = pts.shape[0] # no of sampled points along the ray 

    try:
        # step <-- distance between two consequtive cloud points        
        step_distance = (ptscloud[1] - ptscloud[0])[2]

        # 1. iterate over each ray
        for r in range(M):

            # 2. [find_NN]:  d(1st cloudpoints, ray) = ?
            distance = torch.round(pts[r] - ptscloud[0],decimals=3)

            # 3. [find_NN]:  index ~ closest ptsCloud  
            index = torch.round(distance/step_distance)

            # 4. [find_NN]: discretize the i,j,k
            i = (index[0]).to(dtype=torch.int)
            j = (index[1]).to(dtype=torch.int)
            k = (index[2]).to(dtype=torch.int)

            # 5. [find_NN]: d(cloud[index], ray) ~ [0.00, step_distance]
            index = i * 200 + j * 200 * 200 + k
            distance = torch.round(pts[r] - ptscloud[index],decimals=3)

            print(f"[INFO.find_nn]:\t\t  d([{i},{j},{k}],ray) = {distance}, step = {step_distance}, index={index}")
            print("--" * 70)
            
            # 7. [find_NN]: nn_index <--- (i,j,k) append neighbor to list 
            nn_index.append(torch.tensor([i,j,k]))
                
        # 8. return stack of indexes         
        nn_index = torch.stack(nn_index,dim=0)

        print(f"[INFO.find_nn]:\t\t nn_index = {nn_index.shape}")    
        print("--" * 70)

        return nn_index.long()

    except Exception as e:

        print(f"[ERROR.find_nn]:\t\t {e}")
        print("--" * 70)

        return None 

def render_rays_discrete(ray_steps, rays_o, rays_d, N_samples, pt_cloud, rgb_val, sigma_val, T_n):
    """
    ray_steps: the scalar list of size (N_samples) of steps (see TODOs below)
    rays_o: origin of the rays of size (NX3) 
    rays_d: direction of the rays of size (NX3)
    N_samples: number of samples along the ray
    pt_cloud: point cloud of size (KX3)
    rgb_val: rgb values for every point in the  point cloud of size (KX3)
    sigma_val: density values for every point in the point cloud of size (KX1)

    N --> number of rays
    K --> total number of points in the point cloud
    
    TODO: 
    Inside this function:
    1. generate points along the ray via: pts = ray_o + ray_d*ray_steps --> shape (num_rays, number samples along ray, 3)
    2. Find the nearest indices/points for each point along the ray.
    3. render rgb values for rays using the raw2outputs() function

    
    :returns rgb_val_rays: rgb values for the rays of size (NX3)
    """
    N = rays_o.shape[0]     # Number of rays
    rgb_val_rays = None

    ####################################### 1. [Generating Rays]:  r(t) = o + t*d  ###########################################################################
    try:
        print("--" * 70)
        # 1.1 [Expand shape by repeating N times]    :  r(t),o(t):  (1024,3) --> (1024,N_samples,3) and t : (N_samples) --> (N_samples,3)
        rays_d = rays_d.unsqueeze(1).repeat(1,N_samples,1)
        rays_o = rays_o.unsqueeze(1).repeat(1,N_samples,1)
        ray_steps = ray_steps.repeat(3,1).transpose(0,1)

        pts = rays_o + ray_steps * rays_d

        print(f"[INFO.render_rays_discrete.1]:\t Generate rays ...  pts = {pts.shape}")
        print("--" * 70)

    except Exception as e:
        print("--" * 70)
        print(f"[ERROR.render_rays_discrete.1]:\t Cannot Generate rays {e}")
        print("--" * 70)

    ####################################### 2. [Find Nearest Neighbor]:  t <-- t_n (nearest) ∀ t ∈ rays R^n #####################################################
    pts_indices = []  
    for ray in range(N):
        hihi = find_nn(pts[ray],pt_cloud)
        pts_indices.append(hihi)
        print(f"[INFO.render_rays_discrete.2]:\t Map rays to cloud ...  pts = {hihi.shape}")
        break

    pts_indices = torch.stack(pts_indices)
    print(f"[INFO.render_rays_discrete.2]:\t Map rays to cloud ...  pts = {pts_indices.shape}")
    print("--" * 70)

    ###################################### 3. [Render color]: C(r) <-- ∫Tn (1-exp(-phi_n * delta_n)) * c_n #######################################################
    try:
        z_vals = ray_steps
        raw = torch.stack(rgb_val,sigma_val)
        print(f"[INFO.render_rays_discrete]:\t Raw  ...  raw = {raw.shape}, z_vals={z_vals.shape}")
        rgb_val_rays = raw2outputs(raw,z_vals,rays_d)
        print("--" * 70)
    except Exception as e:
        print(f"[ERROR.render_rays_discrete.3]: {e}")
        print("--" * 70)
    return rgb_val_rays

def regularize_rgb_sigma(point_cloud, rgb_values, sigma_values):
    """
    point_cloud: your point cloud of size (KX3).
    rgb_values: rgb values of the points in the point cloud of size (KX3).
    sigma_values:  Sigma values of the points in the point cloud of size (KX1).
    
    K --> The total number of points in your point cloud (200X200X200)
    TODO: 
    Inside this function:
    Implement  regularization terms for rgb and sigma

    :returns 
    l2_rgb: regularization for rgb - scalar
    l2_sigma: regularization for density - scalar 
    """
    l2_rgb, l2_sigma = None, None
    return l2_rgb, l2_sigma

def train():
    K = None
    device = 'cpu'

    '''
    Do not change below parameters !!!!!!!!
    '''
    N_rand = 1024 # number of rays that are use during the training, IF YOU DO NOT HAVE ENOUGH RAM YOU CAN DECREASE IT BUT DO NOT NOT FORGET TO INCREASE THE N_iter!!!!
    precrop_frac = 0.9 # do not change
    start , N_iters = 0, 100000
    N_samples = 200 # numebr of samples along the ray
    precrop_iters = 0
    lrate = 5e-3 # learning rate
    pts_res = 200 # point resolution of the pooint clooud        
    pts_max = 3.725 # boundary of our point cloud
    near = 2.
    far = 6.

    # You can play with this hyperparameters
    lambda_sigma = 1e-3 # regularization for the lambda sigma (see the final loss in the loop)
    lambda_rgb = 1e-3  # regularization for the lambda color (see the final loss in the loop)


    main_folder_name = 'Train_lego' # folder name where the output images, out variables will be estimated 
    # load dataset
    images, poses, render_poses, hwf, i_split = load_blender_data('data/nerf_synthetic/lego', True, 8)
    print('Loaded blender', images.shape, render_poses.shape, hwf)
    i_train, _, _ = i_split
    print('\n i_train: ', i_train)
    # get white background
    images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])


    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])


    # generate point cloud
    pt_cloud, t_linspace, pt_cloud_indices = create_pointcloud(N = pts_res, min_val = -1*pts_max, max_val = pts_max)
    pt_cloud = pt_cloud.to(device)
    save_folder_test = os.path.join('logs', main_folder_name) 
    os.makedirs(save_folder_test, exist_ok=True)
    torch.save(pt_cloud.cpu(), os.path.join(save_folder_test, 'pts_clous.tns'))
    torch.save(torch.tensor(t_linspace), os.path.join(save_folder_test, 't_linspace.tns'))
    torch.save(torch.tensor(pt_cloud_indices).long(), os.path.join(save_folder_test, 'pt_cloud_indices.tns'))

    sigma_val = torch.ones(pt_cloud.size(0), 1).uniform_(0, 0.5).to(device)
    rgb_val = torch.zeros(pt_cloud.size(0), 3).uniform_().to(device)

    # do not make any change
    sigma_val.requires_grad = True
    rgb_val.requires_grad = True

    optimizer = torch.optim.Adam([{'params':sigma_val},
                                 {'params':rgb_val}],
                                 lr=lrate,
                                 betas=(0.9, 0.999))


    for i in trange(start, N_iters):
        # Random from one image
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3,:4]

        if N_rand is not None:
            # 1. for every pixel in the image, get the ray origin and ray direction
            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            # 2. sample N 3D points along the ray in 3D grid 
            if  i < precrop_iters:
                '''
                if this is True, at the  beggining it will sample rays only from the 'center' of the image to avaid bad local minima
                '''
                dH = int(H//2 * precrop_frac)
                dW = int(W//2 * precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {precrop_iters}")                
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            # select final ray_o and ray_d
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            t_vals = torch.linspace(0., 1., steps=N_samples)
            z_vals = near * (1.-t_vals) + far * (t_vals)
            # 3. TODO: implement steps t_n, compute error
            
            # steps δ_n = δ_n+1 = δ_n)
            print("="*70)
            print("[INFO.train]:\t\t sigma_n = " ,sigma_val.shape)
            print("="*70)

            delta_n = z_vals[1:] - z_vals[:-1]
            # delta_n = delta_n.repeat(sigma_val.shape[0],1)

            print("="*70)
            print("[INFO.train]:\t\t delta_n =" ,delta_n.shape)
            print("="*70)

            # steps T_n = exp(-Σσ_n * δ_n)
            T_n =  torch.exp(-torch.sum(sigma_val * delta_n)) 
            print("[INFO.train]:\t\t T_n = ",delta_n.shape)
            print("="*70)
            
            rgb_map = render_rays_discrete(ray_steps = z_vals,
                                           rays_o = rays_o,
                                           rays_d = rays_d,
                                           N_samples = N_samples,
                                           pt_cloud = pt_cloud,
                                           rgb_val = rgb_val,
                                           sigma_val = sigma_val,
                                           T_n = T_n,) # T_n = distance between adjacent samples
            print("--" * 70)
            print(f"[INFO.train]: rgb_map = {rgb_map.shape}, target_s = {target_s.shape}")
            print("--" * 70)
            # Note that the rgb_map MUST have the same shape as the target_s !!!!
            
            # do not make any change          
            optimizer.zero_grad()
            img_loss = img2mse(rgb_map, target_s)
            reg_loss_rgb, reg_loss_sigma = regularize_rgb_sigma(point_cloud = pt_cloud, rgb_values= rgb_val , sigma_values = sigma_val) # DO NOT FORGET TO CHANGE THIS
            loss = img_loss + lambda_rgb*reg_loss_rgb + lambda_sigma*reg_loss_sigma # --> this is the loss we minimize
            psnr = mse2psnr(img_loss)

            loss.backward()
            optimizer.step()

        if i%100==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} loss image: {img_loss.item()}")


        if i%1000==0: # 
            '''
            YOU DONO NOT NEED TO MAKE ANY CHANGE HERE EXCEPT  render_rays_discrete FUNCTION !!!!!
            at 1000-th iteraion the bulldozer should be appeared when trained with the default hyperparameters
            We save some intermediate images.
            The first 100 images are from the training set and the rest are novel views!. To speed up the generation we renderer every 8th pose/image
            '''
            save_folder_test_img = os.path.join('logs', main_folder_name, f"{i:05d}") 
            os.makedirs(save_folder_test_img, exist_ok=True)
            torch.save(rgb_val.detach().cpu(), os.path.join(save_folder_test_img, 'rgb_{:03d}.tns'.format(i)))
            torch.save(sigma_val.detach().cpu(), os.path.join(save_folder_test_img, 'sigma_{:03d}.tns'.format(i)))
            
            for j in trange(0,poses.shape[0], 8):
                
                pose = poses[j, :3,:4]
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)

                chunk = 200
                # N_rand = chunk
                rgb_image = []
                for k in range(int(coords.size(0)/chunk)):
                    select_coords = coords[k*chunk: (k+1)*chunk].long()  # (N_rand, 2)
                    rays_o_batch = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d_batch = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                    
                    t_vals = torch.linspace(0., 1., steps=N_samples)
                    z_vals = near * (1.-t_vals) + far * (t_vals)

                    with torch.no_grad():
                        rgb_map = render_rays_discrete(ray_steps = z_vals,
                                rays_o = rays_o_batch,
                                rays_d = rays_d_batch,
                                N_samples = N_samples,
                                pt_cloud = pt_cloud,
                                rgb_val = rgb_val,
                                sigma_val = sigma_val) # CHANGE THIS
                        
                    rgb_image.append(rgb_map)
                    
                rgb_image = torch.cat(rgb_image)

                rgb_image = rearrange(rgb_image, '(w h) d -> w h d', w = W)


                rgbimage8 = to8b(rgb_image.cpu().numpy())
                filename = os.path.join(save_folder_test_img, '{:03d}.png'.format(j))
                imageio.imwrite(filename, rgbimage8)

if __name__=='__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor') # UNCOMMENT THIS IF YOU NEED TO RUN IT IN GPU

    train()