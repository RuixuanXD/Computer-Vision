import numpy as np
import cv2
def get_disparity(im1, im2, maxDisp, windowSize):
    """
    creates a disparity map from a pair of rectified images im1 and
    im2, given the maximum disparity MAXDISP and the window size WINDOWSIZE.
    """
    # dispM = np.zeros_like(im1, dtype=float)

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    dispM = np.zeros_like(im1)
    minDispM = np.full_like(im1, np.inf)
    mask = np.ones((windowSize, windowSize))
    for d in range(maxDisp + 1):
        translatedIm2 = np.roll(im2, d, axis=1)
        translatedIm2[:, :d] = 255
        currentDispM = cv2.filter2D((im1 - translatedIm2) ** 2, -1, mask)
        mask_update = currentDispM < minDispM
        dispM[mask_update] = d
        minDispM = np.minimum(minDispM, currentDispM)

    return dispM

 