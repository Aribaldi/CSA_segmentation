import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def trainer(train_loader, val_loader, device, model, optimizer, scheduler, criterion, n_epochs, metric, show_plots,
            output_path):
    '''
    Train a model with given training parameters
    :param model: Pure PyTorch or SMP model to train
    :param show_plots: Show train/validation loss and lr changes
    :param output_path: Path to save model weights or whole model itself to
    '''
    train_loss_list = []
    valid_loss_list = []
    iou_score_list = []
    lr_rate_list = []
    valid_loss_min = np.Inf
    criterion.to((device))

    model_name = model.encoder.__class__.__name__ + '-' + model.decoder.__class__.__name__


    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0
        valid_metric = 0.0

        model.train()
        with tqdm(enumerate(train_loader), postfix={'train_loss': 0.0}) as bar:
            for idx, data in bar:
                input = data['image']
                target = data['mask']

                input, target = input.to(device), target.to(device)
                input = input.float()
                optimizer.zero_grad()
                output = model(input)
                output = torch.squeeze(output)
                target = target.type_as(output)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(data)
                bar.set_postfix(ordered_dict={'train_loss': loss.item()})

        model.eval()
        del data, target, idx
        with torch.no_grad():
            with tqdm(enumerate(val_loader), postfix={'valid_loss': 0.0, 'valid_IoU': 0.0}) as bar:
                for idx, data in bar:
                    input = data['image']
                    target = data['mask']

                    input, target = input.to(device), target.to(device)
                    input = input.float()
                    output = model(input)
                    output = torch.squeeze(output)
                    target = target.type_as(output)
                    loss = criterion(output, target)
                    score = metric(output, target)
                    valid_loss += loss.item() * len(data)
                    valid_metric += score.item() * len(data)
                    bar.set_postfix(ordered_dict={"valid_loss": loss.item(), "valid_metric": score.item()})

        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(val_loader.dataset)
        valid_metric = valid_metric / len(val_loader.dataset)


        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        iou_score_list.append(valid_metric)

        lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])

        print('Epoch: {}  Training loss: {:.6f}  Validation loss: {:.6f} Validation IoU Score: {:.6f}'.format(
            epoch, train_loss, valid_loss, valid_metric))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), str(output_path) + model_name + '.pt')
            valid_loss_min = valid_loss
        scheduler.step(train_loss)

    if show_plots:
        xx = range(n_epochs)
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].plot(xx, train_loss_list)
        ax[0].set(xlabel='epochs', ylabel='loss', title='Train loss')
        ax[1].plot(xx, valid_loss_list)
        ax[1].set(xlabel='epochs', ylabel='loss', title='Validation loss')
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(xx, iou_score_list)
        ax.set(xlabel='epochs', ylabel='IoU', title='Validation IoU')
        plt.show()


