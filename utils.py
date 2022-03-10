import numpy as np


def depth_norm(x, maxDepth):
    return maxDepth / x


def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3: images = np.stack((images, images, images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(depth_norm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth


def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(file.reshape(480, 640, 3) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)
