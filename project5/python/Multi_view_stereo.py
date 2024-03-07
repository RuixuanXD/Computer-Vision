import numpy as np
import cv2
import os


def parse_camera_parameters(file_path):
    file = open('../data/templeR_par.txt', 'r')
    # Read the contents
    camera_params_str = file.read()
        
    # Close the file
    file.close()
    
    # Split the string into lines for each image
    lines = camera_params_str.strip().split('\n')

    # Dictionary to hold the camera parameters for each image
    camera_parameters = {}

    # Process each line
    for line in lines:
        parts = line.split()
        imgname = parts[0]
        # Intrinsic matrix (K)
        K = np.array([
            [float(parts[1]), float(parts[2]), float(parts[3])],
            [float(parts[4]), float(parts[5]), float(parts[6])],
            [float(parts[7]), float(parts[8]), float(parts[9])]
        ])

        # Extrinsic matrix (R|t)
        R = np.array([
            [float(parts[10]), float(parts[11]), float(parts[12])],
            [float(parts[13]), float(parts[14]), float(parts[15])],
            [float(parts[16]), float(parts[17]), float(parts[18])]
        ])
        t = np.array([
            [float(parts[19])],
            [float(parts[20])],
            [float(parts[21])]
        ])

        # Combine R and t to form the extrinsics
        Rt = np.concatenate((R, t), axis=1)

        # Compute the projection matrix as P = K[R|t]
        P = K.dot(Rt)

        # Store the projection matrix with the corresponding image name
        camera_parameters[imgname] = P

    return camera_parameters


def construct_projection_matrix(camera_parameters):
    # Construct and return the projection matrix using the camera parameters
    pass

def get_3d_coord(q, projection_matrix, depth):
    # Implement the Get3dCoord function to compute the 3D coordinate
    pass

def compute_depth_range(bounding_box, projection_matrix):
    # Compute and return min_depth, max_depth based on the bounding box corners' projection
    pass

def compute_consistency(image0, image1, points_3d):
    # Compute and return the consistency score between two images
    pass

def normalized_cross_correlation(color0, color1):
    # Compute and return the normalized cross correlation between two color sets
    pass

def main():
    # Main function to orchestrate the depth map reconstruction
    camera_params = parse_camera_parameters('templrR_par.txt')
    projection_matrix = construct_projection_matrix(camera_params['I0'])
    bounding_box = [-0.023121, -0.038009, -0.091940, 0.078626, 0.121636, -0.017395]
    min_depth, max_depth = compute_depth_range(bounding_box, projection_matrix)
    
    # Load your images here
    I0 = cv2.imread('../data/templeR0013.png')
    I1 = cv2.imread('../data/templeR0014.png')
    I2 = cv2.imread('../data/templeR0016.png')
    I3 = cv2.imread('../data/templeR0043.png')
    I4 = cv2.imread('../data/templeR0045.png')
    
    depth_map = np.zeros_like(I0)  # Assuming grayscale for simplicity
    
    for x in range(I0.shape[1]):
        for y in range(I0.shape[0]):
            if is_background(I0[y, x]):
                continue
            
            best_depth = None
            best_score = -np.inf
            
            for d in np.arange(min_depth, max_depth, depth_step):
                points_3d = [get_3d_coord((x, y), projection_matrix, d)]
                score01 = compute_consistency(I0, I1, points_3d)
                score02 = compute_consistency(I0, I2, points_3d)
                score03 = compute_consistency(I0, I3, points_3d)
                average_score = np.mean([score01, score02, score03])
                
                if average_score > best_score:
                    best_depth = d
                    best_score = average_score
            
            depth_map[y, x] = best_depth if best_score > score_threshold else 0
    
    # Save or display the depth map
    save_depth_map(depth_map, 'depth_map.png')

if __name__ == "__main__":
    main()
