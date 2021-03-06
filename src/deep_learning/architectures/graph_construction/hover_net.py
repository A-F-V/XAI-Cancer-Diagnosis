from src.deep_learning.metrics.dice2 import DICE2
from src.deep_learning.metrics.aji import AJI
from torch import nn, optim, Tensor
import torch
import numpy as np
from src.deep_learning.architectures.components.residual_unit import ResidualUnit
from src.deep_learning.architectures.components.dense_decoder_unit import DenseDecoderUnit
import pytorch_lightning as pl
from src.deep_learning.metrics.confusion_matrix import confusion_matrix
from src.deep_learning.metrics.hover_net_loss import HoVerNetLoss
import matplotlib.pyplot as plt
from PIL import Image
import io
import mlflow
from src.utilities.tensor_utilties import reset_ids
from src.vizualizations.image_viz import plot_images
from src.vizualizations.cellseg_viz import cell_segmentation_sliding_window_gif_example
from src.utilities.img_utilities import tensor_to_numpy
from torch.nn.functional import binary_cross_entropy
import os
from src.deep_learning.metrics.panoptic_quality import panoptic_quality
from src.transforms.cell_segmentation.hovernet_post_processing import assign_instance_class_label, hovernet_post_process
from src.algorithm.pair_mask_assignment import assign_predicted_to_ground_instance_mask
resnet_sizes = [18, 34, 50, 101, 152]


# todo consider using ModuleList instead of Sequential?


class HoVerNet(pl.LightningModule):
    """HoVerNet Architecture (without the Nuclei Classification Branch) as described in:
        Graham, Simon, et al. "Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images." Medical Image Analysis 58 (2019): 101563.

    """

    def __init__(self, num_batches=0, train_loader=None, val_loader=None, categories=False, **kwargs):
        super(HoVerNet, self).__init__()
        resnet_size = kwargs["RESNET_SIZE"] if "RESNET_SIZE" in kwargs else 50
        assert resnet_size in resnet_sizes
        decodersize = (resnet_sizes.index(resnet_size)+2)*0.25
        self.encoder = HoVerNetEncoder(resnet_size)
        self.np_branch = nn.Sequential(HoVerNetDecoder(decodersize), HoVerNetBranchHead("np"))
        self.hover_branch = nn.Sequential(HoVerNetDecoder(decodersize), HoVerNetBranchHead("hover"))
        self.learning_rate = kwargs["START_LR"] if "START_LR" in kwargs else 1e-3
        self.num_batches = num_batches
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = kwargs
        self.args = kwargs
        self.num_batches = num_batches

        self.categories = categories
        if categories:
            self.category_branch = nn.Sequential(HoVerNetDecoder(decodersize), HoVerNetBranchHead("nc"))
   # def setup(self, stage=None):
   #     self.logger.experiment.log_params(self.args)

    def forward(self, sample):
        latent = self.encoder(sample)
        semantic_mask = self.np_branch(latent)
        hover_maps = self.hover_branch(latent)
        if self.categories:
            categories = self.category_branch(latent)
            return semantic_mask, hover_maps, categories
        return semantic_mask, hover_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5, weight_decay=0)
        if self.args["ONE_CYCLE"]:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args['MAX_LR'], total_steps=self.num_batches,  three_phase=True)
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        else:
            return optimizer

    def on_train_start(self):
        self.logger.log_hyperparams(self.args)

    def training_step(self, train_batch, batch_idx):
        i, sm, hv = train_batch['image'].float(), train_batch['semantic_mask'].float(), train_batch['hover_map'].float()
        y = (sm, hv)
        if self.categories and 'category_mask' in train_batch:
            c = train_batch['category_mask'].float()
            y = (sm, hv, c)
        y_hat = self(i)

        loss = HoVerNetLoss()(y_hat, y)

        self.log("train_loss", loss)
        self.log("train_ce_loss", binary_cross_entropy(y_hat[0], y[0]))

        self.train_sample = {"image": i[0],
                             "semantic_mask_ground": sm[0],
                             "hover_map_ground": hv[0],
                             "semantic_mask_pred": y_hat[0][0],
                             "hover_map_pred": y_hat[1][0]}
        return loss

    def validation_step(self, val_batch, batch_idx):
        i, sm, hv, inm = val_batch['image'].float(), val_batch['semantic_mask'].float(
        ), val_batch['hover_map'].float(), val_batch['instance_mask']
        batch_size = i.shape[0]

        y = (sm, hv)
        if self.categories and 'category_mask' in val_batch:
            c = val_batch['category_mask'].float()
            y = (sm, hv, c)

        y_hat = self(i)

        cf = torch.zeros(5, 5)

        m_aji, m_dice, m_sq, m_dq, mpq = 0, 0, 0, 0, 0
        for i in range(batch_size):
            sm_gt, hv_gt = sm[i].squeeze().cpu(), hv[i].squeeze().cpu()
            sm_pred, hv_pred = y_hat[0][i].squeeze().cpu(), y_hat[1][i].squeeze().cpu()
            instance_pred = hovernet_post_process(sm_pred, hv_pred, h=0.5, k=0.5)
            instance_ground = hovernet_post_process(sm_gt, hv_gt, h=0.5, k=0.5)

            # evaluating classification
            if self.categories and 'category_mask' in val_batch:
                c_gt, c_pred = c[i].squeeze().cpu(), y_hat[2][i].squeeze().cpu()
                cell_cat_pred = torch.as_tensor(assign_instance_class_label(instance_pred, c_pred))
                cell_cat_ground = torch.as_tensor(assign_instance_class_label(instance_ground, c_gt))
                matches = list(assign_predicted_to_ground_instance_mask(instance_ground, instance_pred))
                gt_id, pred_id = list(map(lambda x: x[0], matches)), list(map(lambda x: x[1], matches))
                cf += confusion_matrix(cell_cat_ground[gt_id], cell_cat_pred[pred_id], 5)

            # evaluating segmentation
            sq, dq, pq = panoptic_quality(instance_ground, instance_pred)
            aji, dice = AJI(instance_ground, instance_pred), DICE2(instance_ground, instance_pred)
            m_aji, m_dice, m_sq, m_dq, mpq = m_aji+aji, m_dice+dice, m_sq+sq, m_dq+dq, mpq+pq

        m_aji, m_dice, m_sq, m_dq, mpq = m_aji/batch_size, m_dice/batch_size, m_sq/batch_size, m_dq/batch_size, mpq/batch_size
        loss = HoVerNetLoss()(y_hat, y)
        self.log("val_loss", loss)
        return {'loss': loss, 'confusion_matrix': cf, 'AJI': m_aji, 'DICE': m_dice, 'SQ': m_sq, 'DQ': m_dq, 'PQ': mpq}

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def on_train_epoch_end(self):
        pass
        # sm, hv, sm_hat, hv_hat = (self.train_sample["semantic_mask_ground"],
        #                          self.train_sample["hover_map_ground"],
        #                          self.train_sample["semantic_mask_pred"],
        #                          self.train_sample["hover_map_pred"])
        # create_diagnosis((sm.detach().cpu(), hv.detach().cpu()),
        #                 (sm_hat.detach().cpu(), hv_hat.detach().cpu()), self.#current_epoch)

    def validation_epoch_end(self, outputs):
        cf = torch.stack(list(map(lambda x: x['confusion_matrix'], outputs))).sum(dim=0)
        assert cf.shape == (5, 5)
        print(cf)

        aji_v, aji_m = torch.var_mean(torch.as_tensor(list(map(lambda x: x['AJI'], outputs))), unbiased=True)
        dice_v, dice_m = torch.var_mean(torch.as_tensor(list(map(lambda x: x['DICE'], outputs))), unbiased=True)
        pq_v, pq_m = torch.var_mean(torch.as_tensor(list(map(lambda x: x['PQ'], outputs))), unbiased=True)
        sq_v, sq_m = torch.var_mean(torch.as_tensor(list(map(lambda x: x['SQ'], outputs))), unbiased=True)
        dq_v, dq_m = torch.var_mean(torch.as_tensor(list(map(lambda x: x['DQ'], outputs))), unbiased=True)

        self.log("mean_AJI", aji_m)
        self.log("var_AJI", aji_v)
        self.log("mean_DICE", dice_m)
        self.log("var_DICE", dice_v)
        self.log("mean_SQ", sq_m)
        self.log("var_SQ", sq_v)
        self.log("mean_DQ", dq_m)
        self.log("var_DQ", dq_v)
        self.log("mean_PQ", pq_m)
        self.log("var_PQ", pq_v)

    def on_validation_epoch_end(self):
        if self.current_epoch != 0:
            if self.args["EPOCHS"] < 20 or self.current_epoch % ((self.args["EPOCHS"]+20)//10) == 0:

                sample = self.val_dataloader().dataset[0]
                gif_diag_path = os.path.join("experiments", "artifacts", "cell_seg_img.gif")
                cell_segmentation_sliding_window_gif_example(self, sample, gif_diag_path)
                self.logger.experiment.log_artifact(
                    local_path=gif_diag_path, artifact_path=f"Cell_Seg_{self.current_epoch}", run_id=self.logger.run_id)  # , "sliding_window_gif")


def create_diagnosis(y, y_hat, id):
    sm, sm_hat = y[0], y_hat[0]
    sm_hat_hard = (sm_hat > 0.5).int()
    plt.figure()
    plot_images([tensor_to_numpy(sm), tensor_to_numpy(sm_hat_hard),
                tensor_to_numpy(sm_hat)], dimensions=(1, 3), cmap="gray")
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    mlflow.log_image(im, f"{id}_semantic_mask.png")
    plt.close()

    hv_map, hv_map_hat = y[1], y_hat[1]
    print(hv_map.min(), hv_map.max())
    plt.figure()
    plot_images([hv_map[0], hv_map_hat[0], hv_map[1],
                hv_map_hat[1]], dimensions=(2, 2), cmap="jet")
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    mlflow.log_image(im, f"{id}_hover_maps.png")
    plt.close()


def create_resnet_conv_layer(resnet_size, depth):
    """Creates the conv layer for the resnet. Specified in https://iq.opengenus.org/content/images/2020/03/Screenshot-from-2020-03-20-15-49-54.png

    Args:
        resnet_size (int): the canonincal resnet layer (18, 34, 50, 101, 152)
        depth (iny): what layer number (2,3,4,5)

    Returns:
        nn.Sequential : The pytorch layer
    """
    assert resnet_size in resnet_sizes
    assert depth >= 2 and depth <= 5

    times_lookup = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
    first_channel_lookup = [64, 128, 256, 512]

    kernels, channels = None, None
    stride = 1 if depth == 2 else 2
    if resnet_size < 50:
        kernels = [3, 3]
        channels = (np.array([64, 64])*(2**(depth-2))).tolist()
    else:
        kernels = [1, 3, 1]
        channels = (np.array([64, 64, 256])*(2**(depth-2))).tolist()
    times = times_lookup[resnet_size][depth-2]
    in_channel = 64 if depth == 2 else (first_channel_lookup[depth-3]*(4 if resnet_size >= 50 else 1))
    out_channel = first_channel_lookup[depth-2]*4 if resnet_size >= 50 else first_channel_lookup[depth-2]

    return nn.Sequential(ResidualUnit(in_channel, channels, kernels, stride), *[ResidualUnit(out_channel, channels, kernels, 1) for _ in range(times-1)])


class HoVerNetEncoder(pl.LightningModule):  # Returns 1024 maps with images down sampled by 8x
    def __init__(self, resnet_size):
        super(HoVerNetEncoder, self).__init__()
        self.resnet_size = resnet_size

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            *[create_resnet_conv_layer(self.resnet_size, depth) for depth in range(2, 6)],
            nn.Conv2d(2048 if resnet_size >= 50 else 512, 1024, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, sample):
        return self.layers(sample)


def create_dense_decoder_blocks(in_channels, times):  # todo
    # because of concat in DenseDecoderUnit, we add 32 channels to the input
    return nn.Sequential(*[DenseDecoderUnit(in_channels+i*32) for i in range(times)])


class HoVerNetDecoder(pl.LightningModule):
    def __init__(self, size):
        super(HoVerNetDecoder, self).__init__()
        self.size = size
        units1, units2 = int(8*size), int(4*size)  # can choose how large you want decoder to be

        self.level1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=5, padding=2),
            create_dense_decoder_blocks(256, units1),
            nn.Conv2d(256+units1*32, 512, kernel_size=1, padding=0),
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=5, padding=2),
            create_dense_decoder_blocks(128, units2),
            nn.Conv2d(128+32*units2, 128, kernel_size=1, padding=0),
        )
        self.level3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5,
                      padding=2, stride=1, dilation=1, bias=False),
            nn.Conv2d(256, 64, kernel_size=1,
                      padding=0, stride=1, dilation=1, bias=False))

    def forward(self, sample):
        out = nn.Upsample(scale_factor=2, mode='nearest')(sample)
        out = self.level1(out)
        out = nn.Upsample(scale_factor=2, mode='nearest')(out)
        out = self.level2(out)
        out = nn.Upsample(scale_factor=2, mode='nearest')(out)
        return self.level3(out)


class HoVerNetBranchHead(pl.LightningModule):
    def __init__(self, branch):
        super(HoVerNetBranchHead, self).__init__()
        assert branch in ["np", "hover", "nc"]

        self.activate = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        if branch == "np":
            self.head = nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=True),
                nn.Sigmoid())
        elif branch == "nc":
            self.head = nn.Sequential(
                # 5 classes and 1 background. All but one encoding
                nn.Conv2d(64, 6, kernel_size=1, padding=0, bias=True),
                nn.Softmax(dim=1))  # First dim is batch
        else:  # todo is there a better activation function then nothing?
            self.head = nn.Sequential(
                nn.Conv2d(64, 2, kernel_size=1, padding=0, bias=True)
                # ,nn.Tanh()
            )

    def forward(self, sample):
        return self.head(self.activate(sample))
