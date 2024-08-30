# tutorial_utils.py utility functions for the tutorial
# Author:Itzik Ben Sabat sitzikbs[at]gmail.com
# If you use this code,see LICENSE.txt file and cite our work

import numpy as np
import torch
import scipy.interpolate
import scipy.spatial as spatial

def bounding_box_naive(points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    print (points.shape)
    bot_left_x = min(point[0] for point in points)
    bot_left_y = min(point[1] for point in points)
    bot_left_z = min(point[2] for point in points)
    top_right_x = max(point[0] for point in points)
    top_right_y = max(point[1] for point in points)
    top_right_z = max(point[2] for point in points)
    
    return [(bot_left_x, bot_left_y, bot_left_z), (top_right_x, top_right_y, top_right_z)]


class SinglePointCloudDataset():
    def __init__(self, point_filename, points_per_patch):
        self.points_per_patch = points_per_patch
        self.points = np.loadtxt(point_filename).astype(np.float64)[:, 0:3]
        self.rawPoints = self.points
        self.bbdiag = float(np.linalg.norm(self.points.max(0) - self.points.min(0), 2))
        self.points = (self.points - self.points.mean(0)) / (0.5*self.bbdiag)  # shrink shape to unit sphere
        self.kdtree = spatial.cKDTree(self.points, leafsize = 10, balanced_tree=False)
        self.pc_plot = None
        

    def __getitem__(self, index):
        point_distances, patch_point_inds = self.kdtree.query(self.points[index, :], k=self.points_per_patch)
        rad = max(point_distances)
        patch_points = torch.from_numpy(self.points[patch_point_inds, :])

        # center the points around the query point and scale patch to unit sphere
        patch_points = patch_points - torch.from_numpy(self.points[index, :])
        # patch_points = patch_points / rad

        patch_points, trans = self.pca_points(patch_points)
        return torch.transpose(patch_points, 0, 1), trans, rad


    def __len__(self):
        return self.points.shape[0]

    def pca_points(self, patch_points):
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

class SyntheticPointCloudDataset():
    def __init__(self, n_points, jet_order, points_per_patch=128):
        np.random.seed(42)
        self.points_per_patch = points_per_patch
        self.beta = self.generate_random_beta(jet_order)
        self.points = self.generate_synthetic_example(n_points, jet_order, self.beta)
        self.rawPoints = self.points
        self.gt_normals = self.get_gt_normals(jet_order, self.beta, self.points)
        self.bbdiag = float(np.linalg.norm(self.points.max(0) - self.points.min(0), 2))
        self.points = (self.points - self.points.mean(0)) / (0.5*self.bbdiag)  #
        self.kdtree = spatial.cKDTree(self.points, 10)

    def __getitem__(self, index):
        point_distances, patch_point_inds = self.kdtree.query(self.points[index, :], k=self.points_per_patch)
        rad = max(point_distances)
        patch_points = torch.tensor(self.points[patch_point_inds, :], dtype=torch.float32)
        rawPatchPoints = torch.from_numpy(self.rawPoints[patch_point_inds, :])
        
        # center the points around the query point and scale patch to unit sphere
        patch_points = patch_points - torch.tensor(self.points[index, :], dtype=torch.float32)
        patch_points = patch_points / rad
        # if index == 0:
        #     print (f'index: #{index} patch_points:\n{patch_points[:10]},\nrad:\n{rad}')
        #     print (f'patch_point_inds:\n{patch_point_inds[:20]}')
        # patch_points, trans = self.pca_points(patch_points) # the data is generated aligned to z
        return torch.transpose(patch_points, 0, 1), rad, rawPatchPoints

    def __len__(self):
        return self.points.shape[0]

    def generate_random_beta(self, jet_order):
        """
        generate a random set of jet coefficients
        Args:
            jet_order:

        Returns:

        """
        n_coefficients = int((jet_order +1) * (jet_order + 2) / 2)
        beta = np.expand_dims(np.random.uniform(-1, 1, n_coefficients), -1)
        # synBeta = np.array([0.138002,  0.097007, -0.895565, -0.152359,  0.408740, -0.748792, 0.027175,  0.496049, -0.345331,  0.029370]) 
        # print (beta.shape, synBeta.shape)
        
        # return synBeta.reshape(-1,1)
        return beta

    def generate_synthetic_example(self, n_points, jet_order, beta):
        """
        Generate sample points for a random n-jet of order jet_order
        Args:
            n_points: number of output points
            jet_order: het order

        Returns:
            points: xyz coordinates
        """

        x = np.expand_dims(np.random.uniform(-1.0, 1.0, size=n_points), -1)
        y = np.expand_dims(np.random.uniform(-1.0, 1.0, size=n_points), -1)
        # x = np.expand_dims(np.random.uniform(-0.886377, 0.711147, size=n_points), -1)
        # y = np.expand_dims(np.random.uniform(-0.656122, 0.952223, size=n_points), -1)
        M = self.get_vandermonde(x, y, jet_order)
        z = np.dot(M, beta)
        points = np.concatenate([x, y, z], axis=1)
        return points

    def get_gt_normals(self, jet_order, beta, points):
        """
        computer the surface ground truth normals given the surface coefficients
        Args:
            beta: surface coefficients

        Returns:
            normals: surface normals at every point
        """
        x = np.expand_dims(points[:, 0], -1)
        y = np.expand_dims(points[:, 1], -1)

        if jet_order == 1:
            normals = np.concatenate([-beta[0] * np.ones_like(x), -beta[1] * np.ones_like(x), np.ones_like(x)])
        elif jet_order == 2:
            normals = np.concatenate([-(beta[0] + 2 * beta[2] * x + beta[4] * y),
                           -(beta[1] + 2 * beta[3] * y + beta[4] * x),
                           np.ones_like(x)], axis=1)
        elif jet_order == 3:
            y_2 = y * y
            x_2 = x * x
            xy = x * y
            normals = np.concatenate([-(beta[0] + 2 * beta[2] * x + beta[4] * y + 3 * beta[5] * x_2 +
                             2 *beta[7] * xy + beta[8] * y_2),
                           -(beta[1] + 2 * beta[3] * y + beta[ 4] * x + 3 * beta[6] * y_2 +
                             beta[7] * x_2 + 2 * beta[ 8] * xy),
                           np.ones_like(x)], axis=1)
        elif jet_order == 4:
            # [x, y, x_2, y_2, xy, x_3, y_3, x_2 * y, y_2 * x, x_3 * x, y_3 * y, x_3 * y, y_3 * x, y_2 * x_2
            y_2 = y * y
            x_2 = x * x
            x_3 = x_2 * x
            y_3 = y_2 * y
            xy = x * y
            normals = np.concatenate([-(beta[0] + 2 * beta[2] * x + beta[4] * y + 3 * beta[5] * x_2 +
                             2 * beta[7] * xy + beta[8] * y_2 + 4 * beta[9] * x_3 + 3 * beta[11] * x_2 * y
                             + beta[12] * y_3 + 2 * beta[13] * y_2 * x),
                           -(beta[1] + 2 * beta[3] * y + beta[4] * x + 3 * beta[6] * y_2 +
                             beta[7] * x_2 + 2 * beta[8] * xy + 4 * beta[10] * y_3 + beta[11] * x_3 +
                             3 * beta[12] * x * y_2 + 2 * beta[13] * y * x_2),
                           np.ones_like(x)], axis=1)

        normals = normals / np.linalg.norm(normals, ord=2, axis=1, keepdims=True)
        return normals

    def get_vandermonde(self, x, y, jet_order):
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


def compute_principal_curvatures(beta):
    """
    given the jet coefficients, compute the principal curvatures and principal directions:
    the eigenvalues and eigenvectors of the weingarten matrix
    :param beta: batch of Jet coefficients vector
    :return: k1, k2, dir1, dir2: batch of principal curvatures and principal directions
    """
    with torch.no_grad():
        if beta.shape[1] < 5:
            raise ValueError("Can't compute curvatures for jet with order less than 2")
        else:
            b1_2 = torch.pow(beta[:, 0], 2)
            b2_2 = torch.pow(beta[:, 1], 2)
            #first fundemental form
            E = (1 + b1_2).view(-1, 1, 1)
            G = (1 + b2_2).view(-1, 1, 1)
            F = (beta[:, 1] * beta[:, 0]).view(-1, 1, 1)
            I = torch.cat([torch.cat([E, F], dim=2), torch.cat([F, G], dim=2)], dim=1)
            # second fundemental form
            norm_N0 = torch.sqrt(b1_2 + b2_2 + 1)
            e = (2*beta[:, 2] / norm_N0).view(-1, 1, 1)
            f = (beta[:, 4] / norm_N0).view(-1, 1, 1)
            g = (2*beta[:, 3] / norm_N0).view(-1, 1, 1)
            II = torch.cat([torch.cat([e, f], dim=2), torch.cat([f, g], dim=2)], dim=1)

            M_weingarten = -torch.bmm(torch.inverse(I), II)

            curvatures, dirs = torch.symeig(M_weingarten, eigenvectors=True)
            dirs = torch.cat([dirs, torch.zeros(dirs.shape[0], 2, 1, device=dirs.device)], dim=2) # pad zero in the normal direction

    return curvatures, dirs


def normal2rgb(normals):
    '''
    map normal vectors to RGB cube
    Args:
        normals : normal vectors

    Returns:
        mapped normals to rgb
    '''
    r = np.clip(np.expand_dims((127.5 + 127.5 * normals[:, 0]) / 255, -1), 0, 1)
    g = np.clip(np.expand_dims((127.5 + 127.5 * normals[:, 1]) / 255, -1), 0, 1)
    b = np.clip(np.expand_dims((127.5 + 127.5 * normals[:, 2]) / 255, -1), 0, 1)
    return np.concatenate([r, g, b], 1)


def curvatures2rgb(curvatures,  k1_range=[-1, 1], k2_range=[-1, 1]):
    '''
    map principal curvature to RGB values
    Args:
        curvatures : principal vectors
        k1_range : range of maximal principal curvatures
        k2_range : range of minimal principal curvatures
    Returns:
        mapped normals to rgb
    '''
    k1min = k1_range[0]
    k1max = k1_range[1]
    k2min = k2_range[0]
    k2max = k2_range[1]
    mapped_curvatures = curvatures
    mapped_curvatures[mapped_curvatures[:, 0] > k1max, 0] = k1max
    mapped_curvatures[mapped_curvatures[:, 0] < k1min, 0] = k1min
    mapped_curvatures[mapped_curvatures[:, 1] > k2max, 1] = k2max
    mapped_curvatures[mapped_curvatures[:, 1] < k2min, 1] = k2min

    red_dist = np.array([[0, 0.5, 0],  [0.5, 1, 1], [0, 1, 1]])
    green_dist = np.array([[0,  1, 1], [1, 1, 1],   [1, 1, 0]])
    blue_dist = np.array([[1, 1, 0],   [1, 1, 0.5], [0, 0.5, 0]])

    Xq, Yq = mapped_curvatures[:, 0],  mapped_curvatures[:, 1]

    kx, ky = 1, 1 #interpulation params

    red_mapper = scipy.interpolate.RectBivariateSpline(np.array([k1min, 0, k1max]), np.array([k2min, 0, k2max]),
                                                       red_dist, kx=kx, ky=ky)
    green_mapper = scipy.interpolate.RectBivariateSpline(np.array([k1min, 0, k1max]), np.array([k2min, 0, k2max]),
                                                       green_dist, kx=kx, ky=ky)
    blue_mapper = scipy.interpolate.RectBivariateSpline(np.array([k1min, 0, k1max]), np.array([k2min, 0, k2max]),
                                                       blue_dist, kx=kx, ky=ky)

    r = np.around(red_mapper.ev(Xq, Yq), 4)
    g = np.around(green_mapper.ev(Xq, Yq), 4)
    b = np.around(blue_mapper.ev(Xq, Yq), 4)

    return np.concatenate([np.expand_dims(r, -1), np.expand_dims(g, -1), np.expand_dims(b, -1)], 1)

if __name__ == "__main__":
    # dataset = SyntheticPointCloudDataset(n_points=128, jet_order=1, points_per_patch=128)
    pass


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


def generateDataByParameter(beta, min_x, max_x, min_y, max_y, jet_order = 3):
    n_points = 4096
    x = np.expand_dims(np.random.uniform(min_x, max_x, size=n_points), -1)
    y = np.expand_dims(np.random.uniform(min_y, max_y, size=n_points), -1)
    M = get_vandermonde(x, y, jet_order)
    z = np.dot(M, beta).reshape(-1,1)
    print (x.shape, y.shape, z.shape)
    points = np.concatenate([x, y, z], axis=1)
    return points