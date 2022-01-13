from PIL import ImageFilter
from torchvision.transforms import ToPILImage
from torch.nn.functional import conv2d
from torch import Tensor
import cv2
import torch

sobel_x = Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float()/4
sobel_y = Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0).float()/4


def sobel_old(img_batch: Tensor):
    """Runs the sobel filter on the image batch.

    Args:
        img_batch (Tensor): Batch should be (N,H,W) and float
    Returns:
        tuple: the x and y output components, which are tensors of shape (N,H,W) and float
    """
    img_batch = img_batch.unsqueeze(1).float()
    img_x = conv2d(img_batch, sobel_x, stride=1, padding=1).squeeze()
    img_y = conv2d(img_batch, sobel_y, stride=1, padding=1).squeeze()
    return img_x, img_y


def sobel(img: Tensor):
    assert len(img.shape) == 2
    img_np = img.numpy()
    img_normed = cv2.normalize(img_np, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobel_x = cv2.Sobel(img_normed, cv2.CV_32F, 1, 0, ksize=11)
    sobel_y = cv2.Sobel(img_normed, cv2.CV_32F, 0, 1, ksize=11)
    sx_normed = cv2.normalize(sobel_x, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sy_normed = cv2.normalize(sobel_y, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return torch.as_tensor(sx_normed), torch.as_tensor(sy_normed)
    #img = img.unsqueeze(0).unsqueeze(0)
    #img_x = conv2d(img,sobel_x,stride=1,padding=1)
    #img_y = conv2d(img,sobel_y,stride=1,padding=1)
    # return img_x,img_y