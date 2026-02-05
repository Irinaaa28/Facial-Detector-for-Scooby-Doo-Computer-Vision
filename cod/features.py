from skimage.feature import hog
import numpy as np

def extract_hog_features(patches, hog_params):
    features = []
    for patch in patches:
        hog_feature = hog(patch,
                          orientations=hog_params['orientations'],
                          pixels_per_cell=hog_params["pixels_per_cell"],
                          cells_per_block=hog_params["cells_per_block"],
                          block_norm=hog_params["block_norm"],
                          feature_vector=True)
        features.append(hog_feature)
    return np.array(features)