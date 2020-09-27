import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
from loss import FocalLoss
from model import createDeepLabv3
from trainer import train_model
import datahandler
import argparse
import os
import torch

"""
    Version requirements:
        PyTorch Version:  1.2.0
        Torchvision Version:  0.4.0a0+6b959ee
"""

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_directory", default='./data/', type=str)
parser.add_argument(
    "--exp_directory", default='./out/', type=str)
parser.add_argument("--epochs", default=25, type=int)
parser.add_argument("--batchsize", default=24, type=int)
parser.add_argument("--num_workers", default=20, type=int)

args = parser.parse_args()


bpath = args.exp_directory
data_dir = args.data_directory
epochs = args.epochs
batchsize = args.batchsize
# Create the deeplabv3 resnet101 model which is pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
model = createDeepLabv3(outputchannels=12)
model = torch.nn.DataParallel(model)
model.train()

# Create the experiment directory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)


# Specify the loss function
# criterion = torch.nn.CrossEntropyLoss(reduction='mean')
criterion = FocalLoss(size_average=True)


# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)

# Specify the evalutation metrics
#metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}
metrics = {'f1_score': f1_score}


# Create the dataloader
dataloaders = datahandler.get_dataloader_sep_folder(
    data_dir, imageFolder='original', maskFolder='mask' , batch_size=batchsize, num_workers=args.num_workers)
trained_model = train_model(model, criterion, dataloaders,
                            optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)


# Save the trained model
# torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
torch.save(model, os.path.join(bpath, 'weights.pt'))
