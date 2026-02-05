import numpy as np
from skimage.feature import hog

def detect_faces_in_image(img_gray, detector, window_size, hog_params, threshold):
    detections = []
    scores = []

    H, W = img_gray.shape
    win_w, win_h = window_size
    cell_h, cell_w = hog_params["pixels_per_cell"]

    hog_map = hog(img_gray, 
                  orientations=hog_params["orientations"],
                  pixels_per_cell=hog_params["pixels_per_cell"],
                  cells_per_block=hog_params["cells_per_block"],
                  block_norm=hog_params["block_norm"],
                  feature_vector=False)
    
    n_cells_y, n_cells_x = hog_map.shape[:2]
    win_cells_x = win_w // cell_w - 1
    win_cells_y = win_h // cell_h - 1

    w = detector.coef_.ravel()
    b = detector.intercept_[0]

    for y in range(0, n_cells_y - win_cells_y):
        for x in range(0, n_cells_x - win_cells_x):
            hog_patch = hog_map[y:y + win_cells_y, x:x + win_cells_x].ravel()
            score = np.dot(hog_patch, w) + b
            if score > threshold:
                x_min = x * cell_w
                y_min = y * cell_h
                x_max = x_min + win_w
                y_max = y_min + win_h
                detections.append([x_min, y_min, x_max, y_max])
                scores.append(score)
    return np.array(detections), np.array(scores)