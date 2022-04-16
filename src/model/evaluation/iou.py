import np


def IoU(img_1: np.ndarray, img_2: np.ndarray):
    """Calculates Intersection over Union.

    Args:
        img_1 (np.ndarray): Image 1
        img_2 (np.ndarray): Image 2

    Returns:
        float: IoU
    """
    p, s = (img_1*img_2).sum(), (img_1+img_2).sum()
    return p/(s-p)
