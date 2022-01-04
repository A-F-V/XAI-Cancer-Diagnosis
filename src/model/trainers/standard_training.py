import torch
import tqdm


def train_fn(model, dataloader, optimizer, criterion, args):
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
    count = 0
    for i, batch in enumerate(dataloader):
        i, m = batch['image'], batch['semantic_mask']

        x = i.to(args["DEVICE"])
        y = m.to(args["DEVICE"])  # possibly use of epsilon to avoid log of zero

        y_hat = model(x)
        loss = criterion(y_hat, y)

        torch.cuda.empty_cache()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        count += 1

    return train_loss/count
