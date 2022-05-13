import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import io
import matplotlib.pyplot as plt
from sqlalchemy import desc
from src.utilities.img_utilities import tensor_to_numpy
from src.transforms.cell_segmentation.hover_maps import hover_map
from tqdm import tqdm
import os
from numpy.ma import masked_where
from src.transforms.cell_segmentation.percolation import hollow
from src.transforms.cell_segmentation.hovernet_post_processing import hovernet_post_process
from matplotlib.lines import Line2D


def generate_mask_diagram(model, dataloader, mask_name="semantic_mask", args=None):
    """Creates a diagram contrasting model predictions vs ground truths

    Args:
        model (nn.Module): The model that does the prediction
        dataloader (DataLoader): The dataloader (the one from training works fine)
    """
    f, ax = plt.subplots(2, 3, figsize=(30, 20))
    model.eval()
    for ind, batch in enumerate(dataloader):
        i, m = batch['image'], batch[mask_name]
        x = i.to(args["DEVICE"])
        y_hat = model.predict(x)

        y_hat = y_hat.detach().cpu()
        m = m.detach().cpu()

        ax[0, ind].imshow(m[0].permute(1, 2, 0), cmap='gray')
        ax[1, ind].imshow(y_hat[0].permute(1, 2, 0), cmap='gray')
    return f


def cell_segmentation_sliding_window_gif_example(model, sample, location, amplication=1, fps=10):
    """Generates a gif of the sliding window segmentation of a sample

    Args:
        model (pl.LightningModule): The cell segmentation module
        sample (dict): The result of indexing PanNuke or MoNuSeg for example
        location (str): Path to save gif
        amplication (int, optional): How much to amplifiy the semantic prediction. Defaults to 1.
        fps (int, optional): FPS of gif. Defaults to 10.

    Returns:
        BytesIO: A buffer containing the gif
    """
    model.eval()
    model.cuda()

    x_width = sample['image'].shape[2]
    f, ax = plt.subplots(2, 5, figsize=(20, 10))
    with imageio.get_writer(location, mode='I', fps=fps, format="gif") as writer:
        for x in tqdm(range(0, x_width-64, 2), desc="Generating Prediction GIF"):

            cropped_image_orig = sample['image_original'][:, :64, x:x+64]
            cropped_ins_seg = sample["instance_mask"][:, :64, x:x+64].squeeze()
            cropped_image_trans = sample['image'][:, :64, x:x+64]
            cropped_sm_gt = sample['semantic_mask'].squeeze()[:64, x:x+64]
            ground_hv = hover_map(sample["instance_mask"][:, :64, x:x+64].squeeze())
            cropped_hv_x_gt = ground_hv[0]
            cropped_hv_y_gt = ground_hv[1]

            sm_pred, hv_map_pred = model(cropped_image_trans.unsqueeze(0).cuda())[:2]
            ins_pred = hovernet_post_process(sm_pred.detach().cpu().squeeze(), hv_map_pred.detach().cpu().squeeze())

            ax[0, 0].imshow(tensor_to_numpy(cropped_image_orig.squeeze().detach().cpu()))
            ax[0, 0].set_title("Original Image")
            ax[1, 0].imshow(tensor_to_numpy(cropped_image_trans.squeeze().detach().cpu()))
            ax[1, 0].set_title("Transformed Image")

            ax[0, 1].imshow(cropped_sm_gt.squeeze().detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax[0, 1].set_title("Ground Truth Semantic Mask")
            ax[1, 1].imshow((sm_pred.squeeze().detach().cpu()**amplication).numpy(), cmap="gray", vmin=0, vmax=1)
            ax[1, 1].set_title(f"Predicted Semantic Mask Amplified by {amplication}")

            ax[0, 2].imshow(cropped_hv_x_gt.squeeze().detach().cpu().numpy(), cmap="jet", vmin=-1, vmax=1)
            ax[0, 2].set_title("Ground Truth Hover Map X")
            ax[1, 2].imshow(hv_map_pred.squeeze()[0].detach().cpu().numpy(), cmap="jet", vmin=-1, vmax=1)
            ax[1, 2].set_title("Predicted Hover Map X")

            ax[0, 3].imshow(cropped_hv_y_gt.squeeze().detach().cpu().numpy(), cmap="jet", vmin=-1, vmax=1)
            ax[0, 3].set_title("Ground Truth Hover Map Y")
            ax[1, 3].imshow(hv_map_pred.squeeze()[1].detach().cpu().numpy(), cmap="jet", vmin=-1, vmax=1)
            ax[1, 3].set_title("Predicted Hover Map Y")

            ax[0, 4].imshow(cropped_ins_seg.squeeze().detach().cpu().numpy(), cmap="nipy_spectral")
            ax[0, 4].set_title("Ground Truth Instance Mask")
            ax[1, 4].imshow(ins_pred.squeeze().detach().cpu().numpy(), cmap="nipy_spectral")
            ax[1, 4].set_title("Predicted Instance Mask")

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt_img = imageio.imread(buf)
            writer.append_data(plt_img)


def create_coloured_mask(mask: np.ndarray, colour):
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=2).repeat(3, axis=2)  # introduce new axis and then fill with repeat
    colour = np.asarray(colour)
    coloured_mask = np.zeros_like(mask)+colour
    return coloured_mask*mask


def instance_segmentation_vizualised(img: np.ndarray, instance_seg: np.ndarray, cat_pred: np.ndarray, figsize=(20, 20)):
    """Plots image and the segmentation overlayed on top

    Args:
        img (np.ndarray): Original Image (H,W,3)
        instance_seg (np.ndarray): Instance Segmentation of Image (same size) (H,W)
        cat_pred (np.ndarray): List of cell type predictions. cat_pred[i] is cell i's prediction
    """
    assert img.shape[:2] == instance_seg.shape[:], "Image and instance segmentation must be same size"
    assert len(cat_pred) == instance_seg.max(
    )+1, f"Number of predictions {len(cat_pred)-1} must match number of cells {instance_seg.max()}"
    # neo - red, inflam - green, conn - blue, dead - yell, non-neo - orange,
    colour_scheme = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1.0, 1.0, 0.], [1.0, 0.5, 0.]]
    hl = tensor_to_numpy(hollow(torch.tensor(instance_seg)))

    hollow_masks = [np.isin(hl, np.nonzero(cat_pred == cell_type)) for cell_type in range(0, 5)]
    masks = [create_coloured_mask(hm, colour) for i, colour, hm in zip(range(5), colour_scheme, hollow_masks)]
    final_mask = np.sum(masks, axis=0)

    plt.figure(figsize=figsize)
    plt.imshow(tensor_to_numpy(img))
    plt.imshow(final_mask, alpha=0.5)
    lines = [Line2D([0], [0], color=c, lw=4) for c in colour_scheme]
    plt.legend(lines, ["Neo-Plastic", "Inflammatory", "Connective", "Dead", "Epithelial"])
    plt.axis("off")
