import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from dataset import get_valid_dataloader, get_train_dataloader
from unet import UNet
from util import pred, evaluate, dice_coeff
import numpy as np


EPOCH = 50
LEARNING_RATE = 1e-3
L2_DECAY = 1e-4

if __name__ == '__main__':
    net = UNet()
    train_loader, valid_loader = get_train_dataloader(64), get_valid_dataloader(64)
    optimizer = Adam(params=net.parameters(), lr=LEARNING_RATE, weight_decay=L2_DECAY)
    criterion = nn.NLLLoss2d()
    if torch.cuda.is_available():
        net.cuda()

    # train
    for e in range(EPOCH):
        # iterate over batches
        for idx, (img, label) in enumerate(train_loader):
            img = Variable(img.cuda()) if torch.cuda.is_available() else Variable(img)
            label = Variable(label.cuda()) if torch.cuda.is_available() else Variable(label)
            logits, log_logits = net(img)
            loss = criterion(log_logits)
            # fresh gradients
            optimizer.zero_grad()
            # do backward pass
            loss.backward()
            # update
            optimizer.step()

            if idx % 10 == 0:
                print(loss.data[0])

            if e % 1 == 0:
                # validate
                logits, log_logits = pred(valid_loader, net)
                valid_loss = evaluate(valid_loader, net, criterion)
                pred_labels = np.argmax(logits, axis=1)
                dice = dice_coeff(preds=pred_labels, targets=valid_loader.dataset.labels)
                print(valid_loss, dice)
                torch.save(net.parameters(), 'models/unet.pth')