import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from PIL import Image
from eightpoint import eightpoint
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate
from displayEpipolarF import displayEpipolarF
from epipolarMatchGUI import epipolarMatchGUI

def reproject_and_calculate_mean_distance(P, pts2d, pts3d):
    """
    Reproject 3D points to 2D using camera matrix P, then calculate and return the mean distance
    between reprojected points and original 2D points.
    """
    pts3d_homog = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    pts2d_proj_homog = P @ pts3d_homog.T
    pts2d_proj = pts2d_proj_homog[:2] / pts2d_proj_homog[2]
    distances = np.sqrt(np.sum((pts2d_proj.T - pts2d) ** 2, axis=1))
    min_distance = np.min(distances)
    return min_distance

# Load images and points
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')
pts = np.load('../data/someCorresp.npy', allow_pickle=True).tolist()
pts1 = pts['pts1']
pts2 = pts['pts2']
M = pts['M']


intrinsics = np.load('../data/intrinsics.npy', allow_pickle=True).item()
K1 = intrinsics['K1']
K2 = intrinsics['K2']
templeCoords = np.load('../data/templeCoords.npy', allow_pickle=True).item()['pts1']


# write your code here
R1, t1 = np.eye(3), np.zeros((3, 1))
R2, t2 = np.eye(3), np.zeros((3, 1))

F = eightpoint(pts1, pts2, M)
#print(F)
pts2_temple = epipolarCorrespondence(img1, img2, F, templeCoords)

# displayEpipolarF(img1, img2, F)
# epipolarMatchGUI(img1,img2, F)

E = essentialMatrix(F, K1, K2)
#print(E)
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P1 = K1 @ P1  
P2s = camera2(E)



depths = []
pts3d = []
for i in range(4):
    P2_temp = K2 @ P2s[:, :, i]
    pts3d_temp = triangulate(P1, templeCoords, P2_temp, pts2_temple)
    depth = (pts3d_temp[:, 2] > 0) & ((P2_temp @ np.hstack((pts3d_temp, np.ones((pts3d_temp.shape[0], 1)))).T)[2] > 0)
    depths.append(depth.sum())
    pts3d.append(pts3d_temp)

max_depth = max(depths)
correct_index = depths.index(max_depth)
correct_P2 = P2s[:, :, correct_index]
correct_pts3d = pts3d[correct_index]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(correct_pts3d[:, 0], correct_pts3d[:, 1], correct_pts3d[:, 2])
plt.show()


min_distance = 1e12
min_distance1 = 1e12
min_distance2 = 1e12
for i in range(4):
    P2_c = P2s[:, :, i]
    if np.linalg.det(P2_c[:3, :3]) != 1:
        P2_c = K2 @ P2_c

    pts3d_c = triangulate(P1, pts1, P2_c, pts2)
    x1 = P1 @ np.hstack((pts3d_c, np.ones((pts3d_c.shape[0], 1)))).T
    x2 = P2_c @ np.hstack((pts3d_c, np.ones((pts3d_c.shape[0], 1)))).T
    e = 1e-6 
    x1[:, x1[2, :] > e] /= x1[2, x1[2, :] > e]
    x2[:, x2[2, :] > e] /= x2[2, x2[2, :] > e]
    if np.all(pts3d_c[:, 2] > 0):
        distance1 = np.linalg.norm(pts1 - x1[:2, :].T) / pts3d_c.shape[0]
        distance2 = np.linalg.norm(pts2 - x2[:2, :].T) / pts3d_c.shape[0]
        distance = distance1 + distance2
        if distance < min_distance:
            min_distance = distance
            min_distance1 = distance1
            min_distance2 = distance2


print(f'pts1 error: {min_distance1}')
print(f'pts2 error: {min_distance2}')


# save extrinsic parameters for dense reconstruction
R1, t1 = np.eye(3), np.zeros(3)
R2, t2 = correct_P2[:3, :3], correct_P2[:3, 3]
#print({'R1':R1,'t1':t1,'R2':R2,'t2':t2})
np.save('../results/extrinsics', {'R1': R1, 't1': t1, 'R2': R2, 't2': t2})






