import matplotlib.pyplot as plt

import imageio
import io
import matplotlib.pyplot as plt
from src.utilities.img_utilities import tensor_to_numpy
from src.transforms.graph_construction.hover_maps import hover_map
from tqdm import tqdm


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
    f, ax = plt.subplots(2, 4, figsize=(20, 10))
    with imageio.get_writer(location, mode='I', fps=fps, format="gif") as writer:
        for x in tqdm(range(0, x_width-64, 2)):

            cropped_image_orig = sample['image_original'][:, :64, x:x+64]
            cropped_image_trans = sample['image'][:, :64, x:x+64]
            cropped_sm_gt = sample['semantic_mask'].squeeze()[:64, x:x+64]
            ground_hv = hover_map(sample["instance_mask"][:, :64, x:x+64].squeeze())
            cropped_hv_x_gt = ground_hv[0]
            cropped_hv_y_gt = ground_hv[1]

            sm_pred, hv_map_pred = model(cropped_image_trans.unsqueeze(0).cuda())

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

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt_img = imageio.imread(buf)
            writer.append_data(plt_img)
