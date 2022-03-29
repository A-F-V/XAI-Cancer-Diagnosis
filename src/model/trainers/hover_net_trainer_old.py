from src.model.evaluation.hover_net_loss import HoVerNetLoss
from src.model.trainers.base_trainer import Base_Trainer

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomApply
from src.datasets.MoNuSeg import MoNuSeg
from src.transforms.image_processing.augmentation import *
import mlflow
import matplotlib.pyplot as plt
import io
from PIL import Image
from src.vizualizations.cellseg_viz import generate_mask_diagram
from src.datasets.PanNuke import PanNuke
from src.model.architectures.graph_construction.hover_net import HoVerNet
from src.vizualizations.image_viz import plot_images
from src.utilities.img_utilities import tensor_to_numpy


class HoverNetTrainerOLD(Base_Trainer):
    def __init__(self, args):
        super(Base_Trainer, self).__init__()
        self.args = args

    def train(self):
        print("Initializing Training")
        args = self.args
        print(f"The Args are: {args}")

        transforms = Compose([
            Normalize(
                {"image": [0.6441, 0.4474, 0.6039]},
                {"image": [0.1892, 0.1922, 0.1535]}),
            RandomCrop((200, 200)),  # does not work in random apply as will cause batch to have different sized pictures
            RandomApply(
                [
                    # RandomRotate(), - not working #todo! fix
                    RandomFlip()
                    #AddGaussianNoise(0.01, fields=["image"]),
                    #ColourJitter(bcsh=(0.1, 0.1, 0.1, 0.1), fields=["image"])
                ],
                p=0.5)
        ])  # consider adding scale

        device = args["DEVICE"]
        if device == "default" or "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        args["DEVICE"] = device

        print(f"Running on {device}")
        ds = PanNuke(transform=transforms) if args["DATASET"] == "PanNuke" else None
        dl = DataLoader(ds, batch_size=args["BATCH_SIZE"], shuffle=True, num_workers=3)

        model = HoVerNet(self.args["RESNET_SIZE"])
        model.to(device)

        criterion = HoVerNetLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0003)

        scheduler = None
        if args["ONE_CYCLE"]:
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args['MAX_LR'], steps_per_epoch=len(dl), epochs=args["EPOCHS"],  three_phase=True)

        print("Starting Training")
        with mlflow.start_run(run_name=args["RUN_ID"]):
            loop = tqdm(range(args["EPOCHS"]))
            mlflow.log_params(args)
            for epoch in loop:
                loss = train_step(model, dl, optimizer, criterion=criterion, args=args, loop=loop, scheduler=scheduler)
                loop.set_postfix_str(f"Loss: {loss}")

        # mlflow.pytorch.save_model(model, "trained_models/cell_seg_v1.pth")


def train_step(model, dataloader, optimizer, criterion, args, loop, scheduler=None):
    """Performs one epoch's training.

    Args:
        model (nn.Module): The model being trained.
        dataloader (DataLoader): The DataLoader used for training.
        optimizer: The pytorch optimizer used
        criterion: The loss function
        args (dict): Additional arguments for training.
    """
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    train_loss = 0
    mving_avg = 1.3
    count = 0
    loop_for_epoch = tqdm(enumerate(dataloader), total=len(dataloader))
    for ind, batch in loop_for_epoch:
        step = loop.n*len(dataloader) + ind
        loop_for_epoch.set_description(f"Step {step}")

        i, sm, hv = batch['image'].float(), batch['semantic_mask'].float(), batch['hover_map'].float()

        x = i.to(args["DEVICE"])
        y1 = sm.to(args["DEVICE"])  # possibly use of epsilon to avoid log of zero
        y2 = hv.to(args["DEVICE"])

        y = (y1, y2)
        y_hat = model(x)

        loss = criterion(y_hat, y)

        torch.cuda.empty_cache()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=step)

        train_loss += loss.item()
        mving_avg = 0.99 * mving_avg + 0.01 * loss.item()
        mlflow.log_metric("Moving Average of Training loss", mving_avg, step=step)
        loop_for_epoch.set_postfix_str(f"Avg Loss: {mving_avg}")
        count += 1
        if step % 300 == 0:
            print("Creating Image")
            create_diagnosis(y, y_hat, step//300+1)

    return train_loss/count


def create_diagnosis(y, y_hat, id):

    sm, sm_hat = y[0][0].cpu(), y_hat[0][0].cpu()
    sm_hat_hard = (sm_hat > 0.5).int()
    plt.figure()
    plot_images([tensor_to_numpy(sm), tensor_to_numpy(sm_hat_hard),
                tensor_to_numpy(sm_hat)], dimensions=(1, 3), cmap="gray")
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    mlflow.log_image(im, f"{id}_semantic_mask.png")
    plt.close()

    hv_map, hv_map_hat = (y[1][0]).cpu().detach().numpy(), (y_hat[1][0]).cpu().detach().numpy()
    print(hv_map.min(), hv_map.max())
    plt.figure()
    plot_images([hv_map[0], hv_map_hat[0], hv_map[1],
                hv_map_hat[1]], dimensions=(2, 2), cmap="jet")
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    mlflow.log_image(im, f"{id}_hover_maps.png")
    plt.close()
