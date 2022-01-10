from PIL import ImageFilter
from torchvision.transforms import ToPILImage
from torch.nn.functional import conv2d
from torch import Tensor

sobel_x = Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float()/4
sobel_y = Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0).float()/4


def sobel(img_batch: Tensor):
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
