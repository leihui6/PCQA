import torch, sys, os
import numpy as np
sys.path.insert(0, './utils')
sys.path.insert(0, './models')
# sys.path.insert(0, '../trained_models')
import DeepFit
import tutorial_utils as tu
import torch
from tqdm import tqdm

# import ipyvolume as ipv
# import ipywidgets as widgets
# from IPython.display import display
# import functools
# import glob
# import open3d as o3d

jet_order_fit = 3
gpu_idx = 0
device = torch.device("cpu" if gpu_idx < 0 else "cuda:%d" % 0)

def testGPU():
    torch.set_printoptions(precision=6,sci_mode=False)
    print(torch.cuda.get_device_name(0))
    ngpu= 1
    localdevice = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print (localdevice)


def run ():
    fileList = ['./data/bunny.xyz']
    
    for filename in tqdm(fileList):
        print (f'# {filename}')
        filename = filename[:filename.find('.xyz')]
        point_cloud_dataset = tu.SinglePointCloudDataset(f'{filename}.xyz', points_per_patch=256)
        dataloader = torch.utils.data.DataLoader(point_cloud_dataset, batch_size=2048, num_workers=4, shuffle=False, pin_memory=False)
        n_points = point_cloud_dataset.points.shape[0]
        # print (f'point length: {n_points}')
        points = point_cloud_dataset.points

        for batchind, data in tqdm(enumerate(dataloader), total = len(dataloader)):
            points = data[0]
            data_trans = data[1]
            scale_radius = data[2]

            points = points.to(device)
            data_trans = data_trans.to(device)
            scale_radius = scale_radius.to(device)
            points, scale_radius = data[0], data[2]
            beta, n_est, neighbors_n_est = DeepFit.fit_Wjet(points, torch.ones_like(points[:, 0]), 
                                                            order=jet_order_fit,
                                                            compute_neighbor_normals=False)
            n_est = n_est.detach().cpu()
            beta = beta.detach().cpu()
            normals = n_est if batchind==0 else torch.cat([normals, n_est], 0)
            if beta.dim() == 1 and jet_order_fit == 3:
                beta = beta.reshape(-1, 10)
            elif beta.dim() == 1 and jet_order_fit == 4:
                beta = beta.reshape(-1, 15)
            betas = beta if batchind==0 else torch.cat([betas, beta], 0)
        newPointWithNormal = np.concatenate([point_cloud_dataset.rawPoints[:, 0:3], normals, betas], axis = 1)
        print (newPointWithNormal.shape)
        saveFilename = f'{filename}_order{jet_order_fit}_normal_beta.txt'
        np.savetxt(saveFilename, newPointWithNormal, fmt='%1.6f')
        print (f'save to {saveFilename}')

if __name__ == '__main__':
    # testGPU()
    run()