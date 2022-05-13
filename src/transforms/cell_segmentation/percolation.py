from torch import Tensor
from tqdm import tqdm


def island_identifier(img, max_islands=254):  # islands are 0, background is 1
    """Instance segments mask image with non overlapping islands.

    Args:
        img (Tensor or Numpy Array): The bit map image. Islands are 0, background is 1.

    Returns:
        (Tensor or Numpy Array): Background is 0, each island is a different integer from 1 upwards.
    """
    stack = []
    cur_col = 2
    img = img.copy()
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row][col] == 0:
                stack.append((row, col))
                while(len(stack) != 0):
                    x, y = stack.pop()
                    if not (x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1]):
                        if img[x][y] == 0:
                            img[x][y] = cur_col
                            stack.append((x - 1, y))
                            stack.append((x + 1, y))
                            stack.append((x, y - 1))
                            stack.append((x, y + 1))
                cur_col += 1
                assert cur_col <= max_islands
            #print(x, y, cur_col)
    return img - 1  # background is 0, islands are colours from 1 onwards


def hollow(img: Tensor):
    """Hollows out an Instance Mask

    Args:
        img (Tensor): Instance Mask (H,W)

    Returns:
        Tensor: The hollowed out instance mask
    """
    output = img.clone()

    def land_locked(r, c):
        if r <= 0 or r >= (img.shape[0]-1) or c <= 0 or c >= (img.shape[1]-1):
            return False
        return (img[r, c] == img[r, c+1] == img[r, c-1] == img[r+1, c] == img[r-1, c]).item() & (img[r, c] != 0)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if(land_locked(row, col)):
                output[row, col] = 0

    return output
