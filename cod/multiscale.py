import cv2 as cv

def pyramid(image, scale = 0.9, min_size = (64, 64)):
    yield image, 1.0
    current_scale = 1.0

    while True:
        current_scale *= scale
        new_w = int(image.shape[1] * current_scale)
        new_h = int(image.shape[0] * current_scale)
        if new_w < min_size[1] or new_h < min_size[0]:
            break
        resized = cv.resize(image, (new_w, new_h))
        yield resized, current_scale