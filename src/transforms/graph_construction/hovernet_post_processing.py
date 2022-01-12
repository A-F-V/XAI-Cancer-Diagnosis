from PIL import ImageFilter
from torchvision.transforms import ToPILImage
from torch.nn.functional import conv2d
from torch import Tensor
from src.transforms.image_processing.filters import sobel
import torch
from skimage.segmentation import watershed
from skimage.feature.peak import peak_local_max
from scipy import ndimage
from tqdm import tqdm


def _S(hv_maps: Tensor):
    """
    Applies the sobel filter to the hover maps, and gets the importance map from the result.
    Args:
        hv_map (Tensor): hover map of shape (N,2,H,W) and float
    Returns:
        Tensor: The importance map (N,H,W)
    """
    hpx = sobel(hv_maps[:, 0].float())[0].abs()
    hpy = sobel(hv_maps[:, 1].float())[1].abs()
    return torch.maximum(hpx, hpy)


def _markers(q: Tensor, Sm: Tensor, h=0.5, k=0.1):
    """Finds the markers for watershedding.
    Args:
        q (Tensor): The soft semantic map prediction (N,H,W)
        Sm (Tensor): The importance map (N,H,W)
        h (float): The threshold for the semantic map
        k (float): The threshold for the importance map
    Returns:
        Tensor: The markers (N,H,W)
    """
    return torch.maximum(torch.zeros_like(q), (q > h).int()-(Sm > k).int())


def _energy(q: Tensor, Sm: Tensor, h=0.5, k=0.1):
    """Finds the energy map for watershedding.
    Args:
        q (Tensor): The soft semantic map prediction (N,H,W)
        Sm (Tensor): The importance map (N,H,W)
        h (float): The threshold for the semantic map
        k (float): The threshold for the importance map
    Returns:
        Tensor: The energy map (N,H,W)
    """
    return (1-(Sm > k).int())*(q > h).int()


def _watershed(dist: Tensor, mark: Tensor, mask: Tensor = None):
    """Performs watershedding for instance segmentation.

    Args:
        dist (Tensor): The energy landscape (H,W)
        mark (Tensor): The markers (H,W)
        mask (Tensor) optional: A hard mask (H,W)

    Returns:
        Tensor: The prediction (H,W)
    """
    lbs = ndimage.label(mark.numpy())[0]
    return torch.as_tensor(watershed(-(dist.numpy()), markers=lbs, mask=(None if mask is None else mask.numpy()))).int()


# todo do you really want to use the hard mask?
def hovernet_post_process(semantic_mask_pred: Tensor, hv_map_pred: Tensor, h=0.5, k=0.5):  # todo test
    """Takes a prediction and performs instance segmentation. (Usually pre-tiled)

    Args:
        semantic_mask_pred (Tensor): The predicted semantic mask for image (H,W)
        hv_map_pred (Tensor): The predicted hover_maps for the image   (2,H,W)
        h (float): The threshold for the semantic map
        k (float): The threshold for the importance map

    Returns:
        Tensor: The cells instance segmented (H,W) (int)
    """
    sm_hard_pred = (semantic_mask_pred > h).int()
    Sm = _S(hv_map_pred.unsqueeze(0))
    mark = _markers(semantic_mask_pred.unsqueeze(0), Sm, h, k).squeeze()
    energy = _energy(semantic_mask_pred.unsqueeze(0), Sm, h, k).squeeze()
    return _watershed(energy, mark, sm_hard_pred)


# times 2, ensures that an whole number of tiles fit in the image
def tiled_hovernet_prediction(model, img, tile_size=32):
    """Creates a prediction of an entire image through tiling

    Args:
        model (HoVerNet): The HoVerNet Model to use
        img (Tensor): The (3,H,W) image to predict
        tile_size (int, optional): The size of the smaller tiles. Defaults to 32.

    Returns:
        tuple: The semantic mask and hover maps for the entire image, slightly smaller than the original image (due to whole number of tiles used)
    """
    model.eval()
    model.cuda()
    assert tile_size % 2 == 0

    # ENSURE WHOLE NUMBER OF TILES FIT
    dim = list(img.shape[1:])
    dim[0] = dim[0]//tile_size*tile_size
    dim[1] = dim[1]//tile_size*tile_size
    img = img[:, :dim[0], :dim[1]]

    final_sm = torch.zeros(dim[0]-tile_size, dim[1]-tile_size)
    final_hv_x = torch.zeros(dim[0]-tile_size, dim[1]-tile_size)
    final_hv_y = torch.zeros(dim[0]-tile_size, dim[1]-tile_size)

    batch_size = 20//((2*tile_size//64)*(2*tile_size//64))
    print(batch_size)
    batch = None
    batch_loc = []

    def add_tiles(batch, batch_loc):
        with torch.no_grad():
            (sm, hv) = model(batch.cuda())
            sm_b, hv_b = sm.cpu(), hv.cpu()
            for (r, c), sm, hv in zip(batch_loc, sm_b, hv_b):
                sm = sm.squeeze()
                assert len(sm.shape) == 2
                assert len(hv.shape) == 3
                #mask = torch.ones_like(sm)
                # if r!=0:
                #    mask[:tile_size//2,:] = 0
                # if r!=last_row:
                #    mask[-tile_size//2:,:] = 0
                # if c!=0:
                #    mask[:,:tile_size//2] = 0
                # if c!=last_col:
                #    mask[:,-tile_size//2:] = 0
                #final_sm[r:r+tile_size*2,c:c+tile_size*2] += sm*mask
                #final_hv_x[r:r+tile_size*2,c:c+tile_size*2] += hv[0]*mask
                #final_hv_y[r:r+tile_size*2,c:c+tile_size*2] += hv[1]*mask
                final_sm[r:r+tile_size, c:c+tile_size] += sm[tile_size//2:-tile_size//2, tile_size//2:-tile_size//2]
                final_hv_x[r:r+tile_size, c:c+tile_size] += hv[0,
                                                               tile_size//2:-tile_size//2, tile_size//2:-tile_size//2]
                final_hv_y[r:r+tile_size, c:c+tile_size] += hv[1,
                                                               tile_size//2:-tile_size//2, tile_size//2:-tile_size//2]

    for row in tqdm(range(0, dim[0]-tile_size, tile_size)):
        for col in range(0, dim[1]-tile_size, tile_size):
            if batch == None:
                batch = img[:, row:row+tile_size*2, col:col+tile_size*2].unsqueeze(0)
                batch_loc.append((row, col))
            else:
                batch = torch.concat([batch, img[:, row:row+tile_size*2, col:col+tile_size*2].unsqueeze(0)], dim=0)
                batch_loc.append((row, col))
                if batch.shape[0] >= batch_size:
                    add_tiles(batch, batch_loc)
                    del batch
                    torch.cuda.empty_cache()
                    batch = None
                    batch_loc = []
    if batch != None:
        add_tiles(batch, batch_loc)
        del batch
        torch.cuda.empty_cache()
    return final_sm, torch.stack([final_hv_x.squeeze(0), final_hv_y.squeeze(0)], dim=0)
