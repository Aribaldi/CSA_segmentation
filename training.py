import torch
import numpy as np
from tqdm import tqdm
from data_extraction import get_dataloader_single_folder, root_path
import segmentation_models_pytorch as smp
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

train = get_dataloader_single_folder(root_path / 'Dataset')['Train']
valid = get_dataloader_single_folder(root_path / 'Dataset')['Valid']

print(len(train))
print(len(valid))
print(train.dataset)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = smp.Unet('efficientnet-b3', classes=1, activation='sigmoid', encoder_weights=None)
model.to(device)


optimizer = AdamW(lr=1e-4, params=model.parameters())
scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)
criterion = smp.utils.losses.DiceLoss(eps=1.0, activation='sigmoid')
metric_smp = [smp.utils.metrics.IoU(threshold=0.5, activation=None)]
metric_default = smp.utils.metrics.IoU(threshold=0.5, activation=None)

#
#
#
# train_epoch = smp.utils.train.TrainEpoch(
#     model,
#     loss=criterion,
#     metrics=metric_smp,
#     optimizer=optimizer,
#     device=device,
#     verbose=True,
# )
#
# valid_epoch = smp.utils.train.ValidEpoch(
#     model,
#     loss=criterion,
#     metrics=metric_smp,
#     device=device,
#     verbose=True,
# )

# max_score = 0
#
# for i in range(0, 20):
#     print('\nEpoch: {}'.format(i))
#     train_logs = train_epoch.run(train)
#     valid_logs = valid_epoch.run(valid)
#
#     if max_score < valid_logs['iou_score']:
#         max_score = valid_logs['iou_score']
#         torch.save(model.state_dict(), 'best_model.pt')
#         print('Model saved')


n_epochs = 20
train_loss_list = []
valid_loss_list = []
iou_score_list = []
lr_rate_list = []
valid_loss_min = np.Inf


for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0
    valid_metric = 0.0

    model.train()
    bar = tqdm(enumerate(train), postfix={'train_loss': 0.0})
    #print(len(bar))
    for idx, data in bar:
        input = data['image']
        target = data['mask']

        input, target = input.cuda(), target.cuda()
        input = input.float()
        optimizer.zero_grad()
        output = model(input)
        #print(output.shape)
        #output = torch.squeeze(output)
        target = target.type_as(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(data)
        bar.set_postfix(ordered_dict={'train_loss': loss.item()})

    model.eval()
    del data, target, idx
    with torch.no_grad():
        bar = tqdm(enumerate(valid), postfix={'valid_loss': 0.0, 'valid_IoU': 0.0})
        for idx, data in bar:
            input = data['image']
            target = data['mask']

            input, target = input.cuda(), target.cuda()
            input = input.float()
            output = model(input)
            #output = torch.squeeze(output)
            target = target.type_as(output)
            loss = criterion(output, target)
            score = metric_default(output, target)
            valid_loss += loss.item() * len(data)
            valid_metric += score.item() * len(data)
            bar.set_postfix(ordered_dict={"valid_loss": loss.item(), "valid_IoU": score.item()})

    train_loss = train_loss / len(train.dataset)
    valid_loss = valid_loss / len(valid.dataset)
    valid_metric = valid_metric / len(valid.dataset)


    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    iou_score_list.append(valid_metric)

    lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])

    print('Epoch: {}  Training dice Loss: {:.6f}  Validation dice loss: {:.6f} IoU Score: {:.6f}'.format(
        epoch, train_loss, valid_loss, valid_metric))

    if valid_loss <= valid_loss_min:
        print('Validation dice loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'model_test.pt')
        valid_loss_min = valid_loss

    scheduler.step(valid_loss)
