import sys  
sys.path.insert(0, './utils')
from pcqa_utils import *

import time, glob
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_squared_error

import argparse
from threading import Thread
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

fileList = [
    # './data/sample.xyz'
    './data/bunny.xyz'
]

class ThreadPCQA(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def local_pcqa(total_thread:int, thread_index:int, sampledIndices):
    _resList, _totalTime = [], []
    _sampledIndices = np.array_split(sampledIndices, total_thread)[thread_index]
    # print (f'{len(_sampledIndices)} points will be calculated, it takes {100*np.around(len(_sampledIndices) / totalN, 3)}%')

    alphaList, skipIndices = [], []
    with tqdm(total = len(_sampledIndices), position = thread_index) as pbar:
        for indice, i in enumerate(tqdm(_sampledIndices)):
            # 1. Searching neighbor points
            stepName[1] = 'Searching neighbor points'
            timeStart = time.time()
            partPoints, _, rad = processPart (kdtree, i, unitPoints, searchPointSize)
            partPoints = partPoints.numpy().T
            alpha, _ = averageDisPoints(partPoints, searchK = 2)
            # np.savetxt('./tmpBug.xyz', partPoints, fmt='%1.6f')
            timeSeg1 = time.time() - timeStart
            timeStart = time.time()
            
            # 2. Obtain Fitting Results
            stepName[2] = 'Obtain Fitting Results'
            n_est = normals[i, :]
            beta =  betas[i, :]
            # print (f'#{i} p:{p} n_est: {n_est} beta:{beta}')
            timeSeg2 = time.time() - timeStart
            timeStart = time.time()

            # 3. Generate Points by the Fitting Results
            stepName[3] = 'Generate Points by the Fitting Results'
            minP, maxP = bounding_box_naive(partPoints)
            synPartPoints = generateDataByParameter(beta, minP[0], maxP[0], minP[1], maxP[1], jet_order = jet_order)
            tree = cKDTree(synPartPoints, leafsize = 10, balanced_tree=False)
            nearestPoints, nearestIndex, nearestDis = [], [], []
            for pp in partPoints:
                dd, ii = tree.query(pp, k=1)
                nearestPoints.append(synPartPoints[ii])
                nearestIndex.append(ii)
                nearestDis.append(dd)
            if np.mean(nearestDis) > 1.5 * alpha:
                skipIndices.append(i)
                continue
            nearestPoints = np.asarray(nearestPoints)
            # print (f'info: nearestPoints: {nearestPoints.shape}')
            # print (f'nearestPoints mrse: {np.sqrt(np.sum((nearestPoints - partPoints) * (nearestPoints - partPoints)))}')
            # # Original，generated Points and the nearest points on the generated points
            # showPoints ([partPoints, synPartPoints, nearestPoints])
            # showPoints ([partPoints, nearestPoints])
            # break
            timeSeg3 = time.time() - timeStart
            timeStart = time.time()

            # 4. Triangulate the Generated Points
            stepName[4] = 'Triangulate the Generated Point'
            tri = Delaunay(synPartPoints[:, 0:2])
            faces = tri.simplices
            # showMesh(synPartPoints, faces)
            timeSeg4 = time.time() - timeStart
            timeStart = time.time()
            
            
            # 5. Calculate the Distance and Angular 
            stepName[5] = 'Calculate the Distance and Angular'
            distances = calculateDistancesMesh(synPartPoints, faces,
                                    np.array(nearestIndex[:1]), np.array(nearestIndex[1:]),
                                    showPath = False)
            vecs = nearestPoints[1:] - nearestPoints[0]
            pThetas = calculateAngleOnSurface(vecs, n_est)

            distributionPoints = pol2cart(distances, pThetas)
            meanDistributionP = np.mean(distributionPoints, axis = 0)
            # plt.scatter(distributionPoints[:, 0], distributionPoints[:, 1], s = 2)
            # plt.scatter(meanDistributionP[0], meanDistributionP[1], s = 20, c='r')
            # plt.scatter(0, 0, s = 20, c='g')
            # plt.annotate('', xy=(meanDistributionP[0], meanDistributionP[1]), 
            #              xytext=(0,0), arrowprops={'width': 2, 'headwidth': 8, 'headlength': 5})
            # plt.show()
            timeSeg5 = time.time() - timeStart
            timeStart = time.time()

            # 6. Calculate the Quality of Current Point
            stepName[6] = 'Calculate the Quality of Current Point'
            qualityListPerPoint = []
            for gridNum in [gridnum_test_0, gridnum_test_1, gridnum_test_2]: # Here calculate 3 grid-based quality
                minX, maxX = np.min(distributionPoints[:, 0]), np.max(distributionPoints[:, 0])
                minY, maxY = np.min(distributionPoints[:, 1]), np.max(distributionPoints[:, 1])
                width, height = (maxX - minX) / gridNum, (maxY - minY) / gridNum
                # transform all distributed points
                MFrame = np.array([
                    [1, 0, -minX],
                    [0, 1, -minY],
                    [0, 0, 1]]
                ).reshape(3,3)
                grids = np.zeros((gridNum,gridNum))
                gridPoints = np.concatenate([distributionPoints.astype(np.float64), 
                                        np.ones((distributionPoints.shape[0], 1))], axis = 1)
                t_gridPoints = np.dot(MFrame, gridPoints.T).T
                for gp in t_gridPoints:
                    gx, gy = gp[0], gp[1]
                    gridRow = gridNum - 1  if int(gy/height) >= gridNum else int(gy/height)
                    gridCol = gridNum - 1 if int(gx/width) >= gridNum else int(gx/width)
                    grids[gridRow, gridCol] += 1
                timeSeg6 = time.time() - timeStart
                timeStart = time.time()
                hull = ConvexHull(gridPoints[:, 0:2])
                nonZeroRatio = np.count_nonzero(grids) / (hull.volume / (width * height))
                nonZeroRatio = 1 if nonZeroRatio > 1 else nonZeroRatio # [0,1] # offset, can be ignored here
                distance = LA.norm(meanDistributionP)  # can be used for other purposes like boundary detection ...
                quality = nonZeroRatio * (alpha) # alpha is the average value of nearest distances in 'partPoitns'
                qualityListPerPoint.append(quality)
                # draw_mapping_result(grids, hull, minX, maxX, minY, maxY, width, height, gridPoints)
            _resList.append(qualityListPerPoint)
            timeSeg6 = time.time() - timeStart
            timeStart = time.time()
            _totalTime.append([timeSeg1, timeSeg2, timeSeg3, timeSeg4, timeSeg5, timeSeg6])
            pbar.update(1)

    return skipIndices, _resList


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PointCloudQualityAssessment")
    parser.add_argument('-tn', '--thread_number', 
        help="The number of thread used for computation.", type=int)
    parser.add_argument('-sn', '--skip_number', 
        help="The number of points to be skiped.", type=int, default = 12)
    args = parser.parse_args()  
    total_thread_number = args.thread_number
    skip_num = args.skip_number

    jet_order = 3 # defacult is 3
    searchPointSize = 256 # default is 256
    gridnum_test_0 = int (np.around(np.sqrt( searchPointSize) - 1)) # 15
    gridnum_test_1 = gridnum_test_0 - 2 # 13
    gridnum_test_2 = gridnum_test_0 - 4 # 11
    print (f'searchPointSize:{searchPointSize}, gridnum_test_0:{gridnum_test_0}')
    stepName = [None] * 7
    for filenameIndice, filename in enumerate(tqdm(fileList)):
        filename = filename[:filename.find('.xyz')]
        print (f'#{filenameIndice}/{len(fileList)} filename: {filename}')
        data = np.loadtxt(f'{filename}_order{jet_order}_normal_beta.txt')
        # x y z nx ny nz betas (0 - 9)
        rawPoints = data[:, 0:3]
        normals = data[:, 3:6]
        betas = data[:, 6:]
        print (f'rawPoints data format: {rawPoints.shape}')
        print (f'normals data format: {normals.shape}')
        print (f'betas data format: {betas.shape}')

        unitPoints = preprocess(rawPoints)
        
        kdtree = cKDTree(unitPoints, leafsize = 10, balanced_tree = False)

        # evenly sampled by 12
        totalN = unitPoints.shape[0]
        sampledIndices = np.linspace(0, totalN - 1, int(totalN / skip_num)).astype(int)
        print (f'Total Point Size to be calculated: {len(sampledIndices)}')
        
        main_skip_list, main_res_list = [], []
        try:
            tlist = []
            for thread_index in range(total_thread_number):
                t = ThreadPCQA(target = local_pcqa, args = (total_thread_number, thread_index, sampledIndices) )
                t.start()
                tlist.append(t)

            for t_i, t in enumerate(tlist):
                curr_skip_indices, curr_res = t.join()
                main_skip_list += curr_skip_indices
                main_res_list += curr_res
                
        except Exception as error:
            print("An error occurred:", error)
        
        arrResList = np.asarray(main_res_list) # should be -1 x 3
        print (f'{len(main_skip_list)}/{len(sampledIndices)} are skipped')
        remainIndices = list(set(sampledIndices).difference(set(main_skip_list)))
        
        tmp = f'{filename}_quality_score_ours.txt'
        np.savetxt(tmp, np.concatenate([rawPoints[remainIndices], arrResList], axis = 1) , fmt='%1.6f')
        print (f'quality score file save to {tmp}')