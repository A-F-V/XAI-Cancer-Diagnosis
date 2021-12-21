import matplotlib.pyplot as plt


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
