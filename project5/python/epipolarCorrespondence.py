import numpy as np
import cv2

def epipolarCorrespondence(im1, im2, F, pts1):
    """
    Args:
        im1:    Image 1
        im2:    Image 2
        F:      Fundamental Matrix from im1 to im2
        pts1:   coordinates of points in image 1
    Returns:
        pts2:   coordinates of points in image 2
    """
    # pts2 = np.zeros_like(pts1)
    # return pts2

    pts2 = []
    window_size = 10
    window = window_size // 2
    im2_width = im2.shape[1]

    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))

    for pt1_h in pts1_h:
        epiline = F @ pt1_h
        x_values = np.arange(window, im2_width - window)
        y_values = -(epiline[0] * x_values + epiline[2]) / epiline[1]

        best_candidate = None
        min_distance = float('inf')

        for x, y in zip(x_values, y_values):

            if y < window or y >= im2.shape[0] - window:
                continue
            x, y = int(x), int(y)
            window_im2 = im2[y-window:y+window+1, x-window:x+window+1]
            x1, y1 = int(pt1_h[0]), int(pt1_h[1])
            window_im1 = im1[y1-window:y1+window+1, x1-window:x1+window+1]
            distance = np.sum(np.abs(window_im1.astype('int') - window_im2.astype('int')))
            if distance < min_distance:
                min_distance = distance
                best_candidate = (x, y)

        pts2.append(best_candidate)

    return np.array(pts2)