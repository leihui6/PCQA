from scipy.stats import ks_2samp
from tqdm.notebook import tqdm
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
import numpy.linalg as la
import numpy as np
import torch
import open3d as o3d 
import pygeodesic
import pygeodesic.geodesic as geodesic
from tqdm.notebook import tqdm
import igl

# import vtk
# from vtk_helpers import *
import gdist # but no path

import colorsys
import random

def preprocess(points):
    bbdiag = float(np.linalg.norm(points.max(0) - points.min(0), 2))
    points = (points - points.mean(0)) / (0.5*bbdiag)  # shrink shape to unit sphere
    return points


def processPart(kdtree, index, points, searchK):
    # print (f'points[index, :]:{points[index, :]}')
    point_distances, patch_point_inds = kdtree.query(points[index, :], k=searchK)
    rad = max(point_distances)
    patch_points = torch.from_numpy(points[patch_point_inds, :])

    # center the points around the query point and scale patch to unit sphere
    patch_points = patch_points - torch.from_numpy(points[index, :])
    # patch_points = patch_points / rad

    patch_points, trans = pca_points(patch_points)
    return torch.transpose(patch_points, 0, 1), trans, rad


def pca_points(patch_points):
    '''
    Args:
        patch_points: xyz points

    Returns:
        patch_points: xyz points after aligning using pca
    '''
    # compute pca of points in the patch:
    # center the patch around the mean:
    pts_mean = patch_points.mean(0)
    patch_points = patch_points - pts_mean
    trans, _, _ = torch.svd(torch.t(patch_points))
    patch_points = torch.mm(patch_points, trans)
    cp_new = -pts_mean  # since the patch was originally centered, the original cp was at (0,0,0)
    cp_new = torch.matmul(cp_new, trans)
    # re-center on original center point
    patch_points = patch_points - cp_new
    return patch_points, trans


def solve_linear_system(XtX, XtY, sub_batch_size=None):
    """
    Solve linear system of equations. use sub batches to avoid MAGMA bug
    :param XtX: matrix of the coefficients
    :param XtY: vector of the
    :param sub_batch_size: size of mini mini batch to avoid MAGMA error, if None - uses the entire batch
    :return:
    """
    if sub_batch_size is None:
        sub_batch_size = XtX.size(0)
    n_iterations = int(XtX.size(0) / sub_batch_size)
    assert sub_batch_size%sub_batch_size == 0, "batch size should be a factor of {}".format(sub_batch_size)
    beta = torch.zeros_like(XtY)
    print(f'info: XtX: {XtX.shape}')
    n_elements = XtX.shape[1]
    for i in range(n_iterations):
        try:
            L = torch.cholesky(XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...], upper=False)
            beta[sub_batch_size * i:sub_batch_size * (i + 1), ...] = \
                torch.cholesky_solve(XtY[sub_batch_size * i:sub_batch_size * (i + 1), ...], L, upper=False)
        except:
            # # add noise to diagonal for cases where XtX is low rank
            eps = torch.normal(torch.zeros(sub_batch_size, n_elements, device=XtX.device),
                               0.01 * torch.ones(sub_batch_size, n_elements, device=XtX.device))
            eps = torch.diag_embed(torch.abs(eps))
            XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...] = \
                XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...] + \
                eps * XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...]
            try:
                L = torch.cholesky(XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...], upper=False)
                beta[sub_batch_size * i:sub_batch_size * (i + 1), ...] = \
                    torch.cholesky_solve(XtY[sub_batch_size * i:sub_batch_size * (i + 1), ...], L, upper=False)
            except:
                beta[sub_batch_size * i:sub_batch_size * (i + 1), ...], _ =\
                    torch.solve(XtY[sub_batch_size * i:sub_batch_size * (i + 1), ...], XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...])
    return beta


def fit_Wjet(points, weights, order=2, compute_neighbor_normals=False):
    """
    Fit a "n-jet" (n-order truncated Taylor expansion) to a point clouds with weighted points.
    We assume that PCA was performed on the points beforehand.
    To do a classic jet fit input weights as a one vector.
    :param points: xyz points coordinates
    :param weights: weight vector (weight per point)
    :param order: n-order of the jet
    :param compute_neighbor_normals: bool flag to compute neighboring point normal vector

    :return: beta: polynomial coefficients
    :return: n_est: normal estimation
    :return: neighbor_normals: analytically computed normals of neighboring points
    """

    neighbor_normals = None
    D, n_points = points.shape

    # compute the vandermonde matrix
    x = points[0, :].unsqueeze(-1)
    y = points[1, :].unsqueeze(-1)
    z = points[2, :].unsqueeze(-1)
    # weights = weights.unsqueeze(-1)

    # handle zero weights - if all weights are zero set them to 1
    print (f'info: weights:{weights.shape}')
    w_vector = weights
    # valid_count = torch.sum(weights > 1e-3, dim=1)
    # w_vector = torch.where(valid_count > 18, weights.view(1, -1),
                            # torch.ones_like(weights, requires_grad=True).view(1, -1)).unsqueeze(-1)
    if order > 1:
        #pre conditioning
        h = (torch.mean(torch.abs(x)) + torch.mean(torch.abs(y))) / 2 # absolute value added from https://github.com/CGAL/cgal/blob/b9e320659e41c255d82642d03739150779f19575/Jet_fitting_3/include/CGAL/Monge_via_jet_fitting.h
        # h = h.reshape(-1, 1)
        # h = torch.mean(torch.sqrt(x*x + y*y), dim=2)
        # return h
        idx = torch.abs(h) < 0.0001
        h = 0.1
        # h = 0.1 * torch.ones(batch_size, 1, device=points.device)
        # print (f'info: h：{h.shape} {h}， n_points: {type(n_points)} {n_points}, x: {x.shape} {x}')
        x = x / h
        y = y / h
        # h = h.unsqueeze(-1)
        h = torch.from_numpy(np.array([h]))

    if order == 1:
        A = torch.cat([x, y, torch.ones_like(x)], dim=1)
    elif order == 2:
        A = torch.cat([x, y, x * x, y * y, x * y, torch.ones_like(x)], dim=1)
        h_2 = h * h
        D_inv = torch.diag_embed(1/torch.cat([h, h, h_2, h_2, h_2, torch.ones_like(h)], dim=0))
    elif order == 3:
        y_2 = y * y
        x_2 = x * x
        xy = x * y
        A = torch.cat([x, y, x_2, y_2, xy, x_2 * x, y_2 * y, x_2 * y, y_2 * x,  torch.ones_like(x)], dim=1)
        h_2 = h * h
        h_3 = h_2 * h
        # print(f'info: h {h.shape} {h_2.shape}')
        # print (f'info test0 {torch.cat([h, h, h_2, h_2, h_2, h_3, h_3, h_3, h_3, torch.ones_like(h)], dim=0).shape}')
        D_inv = torch.diag_embed(1/torch.cat([h, h, h_2, h_2, h_2, h_3, h_3, h_3, h_3, torch.ones_like(h)], dim=0))
    elif order == 4:
        y_2 = y * y
        x_2 = x * x
        x_3 = x_2 * x
        y_3 = y_2 * y
        xy = x * y
        A = torch.cat([x, y, x_2, y_2, xy, x_3, y_3, x_2 * y, y_2 * x, x_3 * x, y_3 * y, x_3 * y, y_3 * x, y_2 * x_2,
                       torch.ones_like(x)], dim=1)
        h_2 = h * h
        h_3 = h_2 * h
        h_4 = h_3 * h
        D_inv = torch.diag_embed(1/torch.cat([h, h, h_2, h_2, h_2, h_3, h_3, h_3, h_3, h_4, h_4, h_4, h_4, h_4, torch.ones_like(h)], dim=0))
    else:
        raise ValueError("Polynomial order unsupported, please use 1 or 2 ")

    print (f'info: w_vector: {w_vector.shape} A: {A.shape}')
    print (f'info:{A.permute(1, 0).shape},  {(w_vector * A).shape}')
    XtX = torch.matmul(A.permute(1, 0), w_vector * A)
    XtY = torch.matmul(A.permute(1, 0), w_vector * z)

    beta = solve_linear_system(XtX, XtY)

    if order > 1: #remove preconditioning
        print (f'info: D_inv:{D_inv.shape}, beta: {beta.shape}')
        beta = torch.matmul(D_inv, beta)
        print (f'info beta:{beta.shape}')

    # print (f'info beta2:{beta[0:2].squeeze(-1).shape}')
    # print (f'info beta3:{torch.ones(1, device=x.device, dtype=beta.dtype).shape}')

    n_est = torch.nn.functional.normalize(
        torch.cat([-beta[0:2].squeeze(-1), torch.ones(1, device=x.device, dtype = beta.dtype)], dim=0), 
        p=2, dim=0)

    print (f'info beta:{beta.shape}')
    if compute_neighbor_normals:
        beta_ = beta.unsqueeze(0).repeat(n_points, 1, 1)
        print (f'info beta_:{beta_.shape}')
        
        if order == 1:
            # print (f'info n_est:{n_est.shape}')
            neighbor_normals = n_est.unsqueeze(0).repeat(n_points, 1)
            # print (f'info neighbor_normals:{neighbor_normals.shape}')
        elif order == 2:
            neighbor_normals = torch.nn.functional.normalize(
                torch.cat([-(beta_[:, 0] + 2 * beta_[:, 2] * x + beta_[:, 4] * y),
                           -(beta_[:, 1] + 2 * beta_[:, 3] * y + beta_[:, 4] * x),
                           torch.ones(n_points, 1, device=x.device)], dim=1), p=2, dim=1)
        elif order == 3:
            neighbor_normals = torch.nn.functional.normalize(
                torch.cat([-(beta_[:, 0] + 2 * beta_[:, 2] * x + beta_[:, 4] * y + 3 * beta_[:, 5] *  x_2 +
                             2 *beta_[:, 7] * xy + beta_[:, 8] * y_2),
                           -(beta_[:, 1] + 2 * beta_[:, 3] * y + beta_[:, 4] * x + 3 * beta_[:, 6] * y_2 +
                             beta_[:, 7] * x_2 + 2 * beta_[:, 8] * xy),
                           torch.ones(n_points, 1, device=x.device)], dim=1), p=2, dim=1)
        elif order == 4:
            # [x, y, x_2, y_2, xy, x_3, y_3, x_2 * y, y_2 * x, x_3 * x, y_3 * y, x_3 * y, y_3 * x, y_2 * x_2
            neighbor_normals = torch.nn.functional.normalize(
                torch.cat([-(beta_[:, 0] + 2 * beta_[:, 2] * x + beta_[:, 4] * y + 3 * beta_[:, 5] * x_2 +
                             2 * beta_[:, 7] * xy + beta_[:, 8] * y_2 + 4 * beta_[:, 9] * x_3 + 3 * beta_[:, 11] * x_2 * y
                             + beta_[:, 12] * y_3 + 2 * beta_[:, 13] * y_2 * x),
                           -(beta_[:, 1] + 2 * beta_[:, 3] * y + beta_[:, 4] * x + 3 * beta_[:, 6] * y_2 +
                             beta_[:, 7] * x_2 + 2 * beta_[:, 8] * xy + 4 * beta_[:, 10] * y_3 + beta_[:, 11] * x_3 +
                             3 * beta_[:, 12] * x * y_2 + 2 * beta_[:, 13] * y * x_2),
                           torch.ones(n_points, 1, device=x.device)], dim=1), p=2, dim=1)

    
    return beta.squeeze(), n_est, neighbor_normals


def bounding_box_naive(points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    # print (f"bounding_box_naive shape:{points.shape}")
    minX, minY, minZ = np.min(points, axis = 0)
    maxX, maxY, maxZ = np.max(points, axis = 0)
    
    return [(minX, minY, minZ), (maxX, maxY, maxZ )]
    
def get_vandermonde(x, y, jet_order):
        """
        Generate Vandermonde matrix
        Args:
            x: x coordinate vector
            y: y coordinate vector
            jet_order: jet order

        Returns:
            M: The Vandermonde matrix
        """
        if jet_order == 1:
            M = np.concatenate([x, y, np.ones_like(x)], axis=1)
        elif jet_order == 2:
            M = np.concatenate([x, y, x * x, y * y, x * y, np.ones_like(x)], axis=1)
        elif jet_order == 3:
            y_2 = y * y
            x_2 = x * x
            xy = x * y
            M = np.concatenate([x, y, x_2, y_2, xy, x_2 * x, y_2 * y, x_2 * y, y_2 * x, np.ones_like(x)], axis=1)
        elif jet_order == 4:
            y_2 = y * y
            x_2 = x * x
            x_3 = x_2 * x
            y_3 = y_2 * y
            xy = x * y
            M = np.concatenate(
                [x, y, x_2, y_2, xy, x_3, y_3, x_2 * y, y_2 * x, x_3 * x, y_3 * y, x_3 * y, y_3 * x, y_2 * x_2,
                 np.ones_like(x)], axis=1)
        else:
            raise ValueError("Unsupported jet order")

        return M


def generateDataByParameter(beta, min_x, max_x, min_y, max_y, n_points = 64, jet_order = 3, method = 1):
    # randomly generate
    if method == 0:
        x = np.expand_dims(np.random.uniform(min_x, max_x, size=n_points), -1)
        y = np.expand_dims(np.random.uniform(min_y, max_y, size=n_points), -1)
        M = get_vandermonde(x, y, jet_order)
        z = np.dot(M, beta).reshape(-1,1)
        points = np.concatenate([x, y, z], axis=1)
        return points
    # evenly generate
    elif method == 1:
        X = np.expand_dims(np.arange(min_x, max_x, (max_x - min_x) / n_points), -1)
        Y = np.expand_dims(np.arange(min_y, max_y, (max_y - min_y) / n_points), -1)
        xx, yy = np.meshgrid(X, Y)
        allPoints = None
        for ix, x in enumerate(xx):
            for iy, y in enumerate(yy):
    #             print (ix.shape, iy.shape)
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                M = get_vandermonde(x, y, jet_order)
                z = np.dot(M, beta).reshape(-1,1)
                oncePoints = np.concatenate([x, y, z], axis=1)
                allPoints = oncePoints if (ix == 0 and iy == 0) else np.concatenate([allPoints, oncePoints], axis=0)
            break
        return allPoints


def calculateDistanceTwoPoints(p1, p2):
    return np.sqrt(np.sum((p1-p2) * (p1-p2)))


def calculateDistancesMesh(points, faces, sourceIndices, targetIndices, showPath = True):
    # without path
    if showPath == False:
        distances = gdist.compute_gdist(points, faces.astype(np.int32), 
                                source_indices=sourceIndices.astype(np.int32), target_indices=targetIndices.astype(np.int32))
        where_are_inf = np.isinf(distances)
        if len(where_are_inf) > 0: distances[where_are_inf] = 0.000001
    else:
        geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)
        sourceIndex = sourceIndices[0]
        path_actor_list, distances = [], []
        targetIndicesList = targetIndices.tolist()
        for i, oneTargeIndex in enumerate(tqdm(targetIndicesList)):
            distance, path = geoalg.geodesicDistance(sourceIndex, oneTargeIndex)
            path_actor = createPolyLineActor(path, color=(0,0,0))
            path_actor_list.append(path_actor)
            distances.append(distance)
        # Create actors
        polydata_actor = createPolyDataActor(polydataFromPointsAndCells(points, faces), 
                                            color = (0, 1, 0), opacity = 0.5)
        point_actors = [createSphereActor(points[indx], radius=0.01) for indx in targetIndices]
        # Show VTK render window
        v = Viewer()
        v.addActors([polydata_actor, *point_actors] + path_actor_list)
        v.show()
    return np.array(distances)


def calcAngle180(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)


def calcAngle360(v1, v2):
    '''
    Supports calculations greater than 180 degrees, the angle is counterclockwise
    :param v1:
    :param v2:
    :return: angle in degree
    '''
    # print(v1, v2)
    r = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2)+0.00001))
    deg = r * 180 / np.pi
    a1 = np.array([*v1, 0])
    a2 = np.array([*v2, 0])
    a3 = np.cross(a1, a2)
    if np.sign(a3[2]) < 0:
        deg = 360 - deg
    return deg


def calculateVecAngle(vec):
    '''
    calculate the angle between first vector with others
    '''
    # normVecList = normVec(vec)
    thetas = []
    baseVec = np.array([1,0])
    for i in range(len(vec)):
        thetas.append(calcAngle360(baseVec, vec[i]))
    thetas = np.array(thetas)
    # thetas = np.sort(thetas)
    # print(thetas)
    # diffThetas = np.diff(thetas)
    return thetas


def normVec(vecList):
    normVecList = []
    for _vec in vecList:
        length = np.linalg.norm(_vec)
        normalized_vector = np.divide(_vec, length)
        normVecList.append(normalized_vector)
    return normVecList


def lscm(v, f, interestIndex = None):
    b = np.array([2, 1])
    bnd = igl.boundary_loop(f)
    b[0] = bnd[0]
    b[1] = bnd[int(bnd.size / 2)]
    bc = np.array([[0.0, 0.0], [1.0, 0.0]])
    # LSCM parametrization
    _, uv = igl.lscm(v, f, b, bc)
    interestedPoints = uv[interestIndex]
    if interestIndex == None:
        return uv
    else:
        return interestedPoints


def calculateAngleMesh(vectices, faces, nearestIndex):
    pointsOnSurfaceLSCM = lscm(vectices, faces, nearestIndex)
    # import matplotlib.pyplot as plt
    # plt.scatter(uv[:, 0], uv[:, 1], s = 1, c ='green')
    # print (pointsOnSurfaceLSCM.shape)
    # plt.scatter(pointsOnSurfaceLSCM[:, 0], pointsOnSurfaceLSCM[:, 1], s = 3, c ='red')
    # for p in pointsOnSurfaceLSCM:
    #     X = [pointsOnSurfaceLSCM[0, :][0], p[0]]
    #     Y = [pointsOnSurfaceLSCM[0, :][1], p[1]]
    #     plt.plot(X, Y, c = 'blue', linewidth=0.5)
    # plt.show()

    return calculateVecAngle(pointsOnSurfaceLSCM[1:] - pointsOnSurfaceLSCM[0])


def projectVecOnSurface(n, vecList):
    projectedVecList = []
    for vec in vecList:
        projectVec = vec + np.dot(n, (np.dot(vec, n) * -1))
        projectedVecList.append(projectVec)
    return np.array(projectedVecList)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y]).T


def find_transformation(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
 
    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions

def calculateAngleOnSurface(vecs, normal):
    pVecs = projectVecOnSurface(normal, vecs)
    baseVec = np.array([0,0,1])
    if np.dot(normal, baseVec) > 0:
        # print ('base vec changed')
        baseVec *= -1
    nM = find_transformation(normal, baseVec)
    t_pVecs = np.dot(nM, pVecs.T).T
    t_pVecs = t_pVecs[:, 0:2]
    pThetas = calculateVecAngle(t_pVecs)
    pThetas = np.deg2rad(pThetas)
    return pThetas


def GeneratePointInCyclePrefect(point_num, radius):
    xyNum = int(np.sqrt(point_num) + np.pi)
    xVec = np.linspace(-radius, radius, xyNum)
    yVec = np.linspace(-radius, radius, xyNum)
    origin = np.array([0,0])
    points = []
    for i, x in enumerate(xVec):
        for j, y in enumerate(yVec):
            p = np.array([x,y])
            d = calculateDistanceTwoPoints(origin, p)
            if d < radius:
                points.append(p)
    points = np.asarray(points)
    return points[:, 0].reshape(-1,1), points[:, 1].reshape(-1,1)


def GeneratePointInCycle1(point_num, radius):
    theta = np.random.uniform(low = 0, high = 2*np.pi, size = point_num)
    k = np.random.uniform(low = 0, high = 1, size = point_num)
    x = np.cos(theta)* (np.sqrt(k)*radius)
    y = np.sin(theta)* (np.sqrt(k)*radius)
    # plt.plot(x, y, '*', color = "black") 
    return x.reshape(-1, 1), y.reshape(-1, 1)


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    return hls_colors
 
 
def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
#         r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([_r, _g, _b])
    return rgb_colors


def showMesh(_vectices, _face):
    import polyscope as ps
    ps.init()
    ps.register_point_cloud("my points", _vectices)
    ps.register_surface_mesh("my mesh", _vectices, _face, smooth_shade=True)
    ps.show()


def showPoints (pList):
    opList = []
    colors = ncolors(len(pList))
    for i, p in enumerate(pList):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p)
        pcd.paint_uniform_color(colors[i]) 
        opList.append(pcd)
    coord = o3d.geometry.TriangleMesh().create_coordinate_frame()
    # o3d.visualization.draw_geometries([coord] + opList)
    o3d.visualization.draw_geometries(opList)


def diffVec(vec1, vec2):
    if len(vec1) > len(vec2):
        return vec1[:len(vec2)] - vec2
    elif len(vec1) < len(vec2) :
        return vec2[:len(vec1)] - vec1
    else:
        return vec1 - vec2
    
    
def asRigidAsPossible(v, f, interestedIndex = None):
    ## Find the open boundary
    bnd = igl.boundary_loop(f)

    ## Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(v, bnd)

    ## Harmonic parametrization for the internal vertices
    uv = igl.harmonic(v, f, bnd, bnd_uv, 1)

    arap = igl.ARAP(v, f, 2, np.zeros(0))
    uva = arap.solve(np.zeros((0, 0)), uv)
    
    if interestedIndex == None:
        return uva
    else:
        return uva[interestedIndex]
    

def averageDisPoints(points, leadSize = 10, searchK = 5):
    '''calculate the average distance among points given a search size K'''
    tree = cKDTree(points, leafsize = leadSize, balanced_tree=False)
    # plt.scatter(standPoints[:, 0], standPoints[:, 1], s=1)
    totalDis = []
    for p in points:
        dd, ii = tree.query(p, k = searchK)
        # print(dd, ii)
        totalDis.append(np.mean(dd[1:]))
    return np.mean(totalDis), np.std(totalDis)
