from PIL import ImageFilter
from numpy import dtype, tile
import numpy as np
from torchvision.transforms import ToPILImage
from torch.nn.functional import conv2d
from torch import Tensor
from src.transforms.image_processing.filters import sobel
import torch
from skimage.segmentation import watershed
from skimage.feature.peak import peak_local_max
from scipy import ndimage
from tqdm import tqdm
from src.transforms.image_processing.augmentation import Normalize
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes, binary_closing
from scipy.ndimage.measurements import label
from skimage.segmentation import watershed
from src.utilities.tensor_utilties import reset_ids
import cv2


def _S(hv_map: Tensor):
    """
    Applies the sobel filter to the hover maps, and gets the importance map from the result.
    Args:
        hv_map (Tensor): hover map of shape (N,2,H,W) and float
    Returns:
        Tensor: The importance map (N,H,W)
    """
    hv_horiz, hv_vert = hv_map
    hpx = sobel(hv_horiz.float())[0].abs()
    # sobel here outputs small numbers for edges and large numbers for cell centres, hence want to invert to get importance (0 for cell centres, 1 for edges)
    hpy = sobel(hv_vert.float())[1].abs()
    return torch.maximum(1-hpx, 1-hpy).squeeze()


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


def hovernet_post_process_old(semantic_mask_pred: Tensor, hv_map_pred: Tensor, h=0.5, k=0.5):
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


def hovernet_post_process(sm: Tensor, hv_map: Tensor, h=0.5, k=0.5, smooth_amt=5):  # todo doc and annotate
    Sm = _S(hv_map)
    thresh_q = (sm > h)
    thresh_q = torch.as_tensor(remove_small_objects(thresh_q.numpy(), min_size=30))
    Sm = (Sm - (1-thresh_q.float())).clip(0)  # importance regions with background haze removed via mask with clipping

    # to get areas of low importance (i.e. centre of cells) as high energy and areas close to border are low energy
    energy = (1-Sm)*thresh_q
    # also clip again background
    energy = torch.as_tensor(cv2.GaussianBlur(energy.numpy(), (smooth_amt, smooth_amt), 0)
                             )  # smooth landscape # especially important for long cells

    markers = (thresh_q.float() - (Sm > k).float())
    markers = label(markers)[0]
    # Slightly different to paper - I use the energy levels instead because they have been smoothed
    markersv2 = (energy > k).numpy()
    markersv2 = binary_fill_holes(markersv2)
    markersv2 = label(markersv2)[0]
    final = watershed(-energy.numpy(), markers=markersv2, mask=thresh_q.numpy())
    final = reset_ids(final)
    return torch.as_tensor(final, dtype=torch.int)

# times 2, ensures that an whole number of tiles fit in the image


def tiled_hovernet_prediction(model, img, tile_size=32):
    """Creates a prediction of an entire image through tiling

    Args:
        model (HoVerNet): The HoVerNet Model to use
        img (Tensor): The (3,H,W) image to predict. This has already been normalized
        tile_size (int, optional): The size of the smaller tiles. Defaults to 32.

    Returns:
        tuple: The semantic mask, hover maps, and cell type for the entire image, slightly smaller than the original image (due to whole number of tiles used)
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
    final_cat = torch.zeros(6, dim[0]-tile_size, dim[1]-tile_size)
    final_hv_x = torch.zeros(dim[0]-tile_size, dim[1]-tile_size)
    final_hv_y = torch.zeros(dim[0]-tile_size, dim[1]-tile_size)

    batch_size = 10//((2*tile_size//64)*(2*tile_size//64))
    batch = None
    batch_loc = []

    def add_tiles(batch, batch_loc):
        with torch.no_grad():
            (sm, hv, cat) = model(batch.cuda())
            sm_b, hv_b, cat_b = sm.cpu(), hv.cpu(), cat.cpu()
            for (r, c), sm, hv, cat in zip(batch_loc, sm_b, hv_b, cat_b):
                sm = sm.squeeze()
                cat = cat.squeeze()

                assert len(cat.shape) == 3
                assert len(sm.shape) == 2
                assert len(hv.shape) == 3
                # mask = torch.ones_like(sm)
                # if r!=0:
                #    mask[:tile_size//2,:] = 0
                # if r!=last_row:
                #    mask[-tile_size//2:,:] = 0
                # if c!=0:
                #    mask[:,:tile_size//2] = 0
                # if c!=last_col:
                #    mask[:,-tile_size//2:] = 0
                # final_sm[r:r+tile_size*2,c:c+tile_size*2] += sm*mask
                # final_hv_x[r:r+tile_size*2,c:c+tile_size*2] += hv[0]*mask
                # final_hv_y[r:r+tile_size*2,c:c+tile_size*2] += hv[1]*mask
                final_sm[r:r+tile_size, c:c+tile_size] += sm[tile_size//2:-tile_size//2, tile_size//2:-tile_size//2]
                final_hv_x[r:r+tile_size, c:c+tile_size] += hv[0,
                                                               tile_size//2:-tile_size//2, tile_size//2:-tile_size//2]
                final_hv_y[r:r+tile_size, c:c+tile_size] += hv[1,
                                                               tile_size//2:-tile_size//2, tile_size//2:-tile_size//2]
                final_cat[:, r:r+tile_size, c:c+tile_size] += cat[:,
                                                                  tile_size//2:-tile_size//2, tile_size//2:-tile_size//2]

    for row in range(0, dim[0]-tile_size, tile_size):
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
    return final_sm, torch.stack([final_hv_x.squeeze(0), final_hv_y.squeeze(0)], dim=0), final_cat


def assign_instance_class_label(instance_map: Tensor, nucleus_prediction: Tensor):
    predictions = [5]
    for nucleus_id in range(1, instance_map.max()+1):
        mask = instance_map == nucleus_id
        nucleus_category_mask = mask*nucleus_prediction
        predictions.append(nucleus_category_mask.sum(dim=(1, 2))[:5].argmax().item())   # add prediction to predictions
    return np.asarray(predictions)


@torch.no_grad()
def instance_mask_prediction_hovernet(model, img, tile_size=128, pre_normalized=False, all_channels=False, h=0.5, k=0.5):
    if pre_normalized:
        t_img = img
    else:
        normalizer = Normalize(
            {"image": [0.6441, 0.4474, 0.6039]},
            {"image": [0.1892, 0.1922, 0.1535]})
        t_img = normalizer({"image": img.clone()})["image"]
    sm_pred, hv_pred, cat_pred = tiled_hovernet_prediction(model, t_img, tile_size)
    ins_pred = hovernet_post_process(sm_pred, hv_pred, h=h, k=k)
    cell_cat_pred = assign_instance_class_label(ins_pred, cat_pred)
    assert len(cell_cat_pred) == ins_pred.max()+1
    return (ins_pred, cell_cat_pred) if not all_channels else (ins_pred, cell_cat_pred, sm_pred, hv_pred)


def cut_img_from_tile(img, tile_size=32):
    dim = list(img.shape[1:])
    dim[0] = dim[0]//tile_size*tile_size
    dim[1] = dim[1]//tile_size*tile_size
    return img.clone()[:, tile_size//2:dim[0]-tile_size//2, tile_size//2:dim[1]-tile_size//2]
