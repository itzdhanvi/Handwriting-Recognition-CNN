import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 28


def preprocess_emnist(image, label):
    """Preprocessing used during CNN training/evaluation."""
    image = tf.cast(image, tf.float32) / 255.0
    # EMNIST orientation correction used consistently at inference.
    image = tf.transpose(image, perm=[1, 0, 2])
    return image, label


def preprocess_user_image(image):
    """
    Convert a user drawing/upload into a centered 28x28 grayscale image.
    Returns shape (1, 28, 28, 1), float32 in [0, 1].
    """
    image = np.array(image)

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = image.astype(np.uint8)

    # Convert to white character on black background.
    if np.mean(image) > 127:
        image = 255 - image

    _, image = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(image)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        image = image[y:y+h, x:x+w]

    h, w = image.shape

    if h == 0 or w == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    image = cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = image

    canvas = canvas.astype(np.float32) / 255.0
    return canvas[np.newaxis, ..., np.newaxis]
