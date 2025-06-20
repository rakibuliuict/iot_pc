# # supervised_train.py
# import os
# import sys
# import argparse
# import random
# import logging
# import numpy as np
# from tqdm import tqdm
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader

# from networks.ResVNet import ResVNet
# from pancreas.Vnet import VNet
# from dataloaders.LADataset import LAHeart
# from utils.LA_utils import to_cuda
# from pancreas.losses import DiceLoss
# from utils import test_3d_patch

# parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str, default='/content/drive/MyDrive/0SSL/Dataset/2018_UTAH_MICCAI')
# parser.add_argument('--list_path', type=str, default='/content/drive/MyDrive/0SSL/WUB_mail/LA_SSL/Datasets/la/data_split')
# parser.add_argument('--exp', type=str, default='Supervised', help='Experiment name')
# parser.add_argument('--model', type=str, default='VNet', choices=['VNet', 'ResVNet'])
# parser.add_argument('--epochs', type=int, default=10)
# parser.add_argument('--batch_size', type=int, default=2)
# parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--gpu', type=str, default='0')
# parser.add_argument('--save_path', type=str, default='/content/drive/MyDrive/0SSL/WUB_mail/model/PreTrained')
# args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# if not os.path.exists(args.save_path):
#     os.makedirs(args.save_path)

# logging.basicConfig(filename=os.path.join(args.save_path, "train_log.txt"),
#                     level=logging.INFO, format='%(asctime)s %(message)s')
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# logging.info(str(args))

# # Fix randomness
# cudnn.benchmark = False
# cudnn.deterministic = True
# torch.manual_seed(1337)
# torch.cuda.manual_seed(1337)
# np.random.seed(1337)
# random.seed(1337)

# # Load dataset
# dataset = LAHeart(args.root_path, args.list_path, split='train')
# dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

# # Model
# if args.model == 'VNet':
#     net = VNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
# else:
#     net = ResVNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)

# net = nn.DataParallel(net).cuda()

# # Loss and optimizer
# ce_loss = nn.CrossEntropyLoss()
# dice_loss = DiceLoss(nclass=2)
# optimizer = optim.Adam(net.parameters(), lr=args.lr)

# # Training
# max_epoch = args.epochs
# best_dice = 0

# for epoch in range(max_epoch):
#     net.train()
#     running_loss = 0.0
#     for batch in tqdm(dataloader, ncols=80):
#         images, labels = batch['image'].cuda(), batch['label'].cuda()
#         outputs, _ = net(images)

#         loss_ce = ce_loss(outputs, labels)
#         loss_dice = dice_loss(outputs, labels)
#         loss = loss_ce + loss_dice

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     logging.info("Epoch [%d/%d], Loss: %.4f" % (epoch + 1, max_epoch, running_loss / len(dataloader)))

#     if (epoch + 1) % 2 == 0:
#         net.eval()
#         dice_score = test_3d_patch.var_all_case_LA(net, num_classes=2, patch_size=(112, 112, 80), stride_xy=18, stride_z=4)
#         logging.info("Validation Dice Score: %.4f" % dice_score)

#         if dice_score > best_dice:
#             best_dice = dice_score
#             best_model_path = os.path.join(args.save_path, f"best_model_epoch_{epoch+1}.pth")
#             torch.save(net.state_dict(), best_model_path)
#             logging.info(f"Saved best model to {best_model_path}")

# logging.info("Training complete. Best Dice Score: %.4f" % best_dice)


# 

# import os
# import sys
# import argparse
# import random
# import logging
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import shutil

# from networks.ResVNet import ResVNet
# from pancreas.Vnet import VNet
# from dataloaders.LADataset import LAHeart
# from utils.LA_utils import to_cuda
# from pancreas.losses import DiceLoss
# from utils import test_3d_patch

# parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str, default='/content/drive/MyDrive/0SSL/Dataset/2018_UTAH_MICCAI')
# parser.add_argument('--list_path', type=str, default='/content/drive/MyDrive/0SSL/WUB_mail/LA_SSL/Datasets/la/data_split')
# parser.add_argument('--exp', type=str, default='Supervised')
# parser.add_argument('--model', type=str, default='VNet', choices=['VNet', 'ResVNet'])
# parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--batch_size', type=int, default=2)
# parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--gpu', type=str, default='0')
# parser.add_argument('--save_path', type=str, default='/content/drive/MyDrive/0SSL/WUB_mail/model/PreTrained')
# args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# if not os.path.exists(args.save_path):
#     os.makedirs(args.save_path)

# log_file = os.path.join(args.save_path, "train_log.txt")
# logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# logging.info(str(args))

# # Reproducibility
# seed = 1337
# cudnn.benchmark = False
# cudnn.deterministic = True
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

# # Dataset
# dataset = LAHeart(args.root_path, args.list_path, split='train')
# dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

# # Model
# if args.model == 'VNet':
#     net = VNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
# else:
#     net = ResVNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
# net = nn.DataParallel(net).cuda()

# # Loss and optimizer
# ce_loss = nn.CrossEntropyLoss()
# dice_loss = DiceLoss(nclass=2)
# optimizer = optim.Adam(net.parameters(), lr=args.lr)

# # === Manual Resume Configuration ===
# start_epoch = 0  # Change this to the desired epoch to resume from
# best_dice = 0
# checkpoint_path = os.path.join(args.save_path, "last_checkpoint.pth.tar")

# if start_epoch > 0 and os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     best_dice = checkpoint.get('best_dice', 0)
#     logging.info(f"Resumed training from epoch {start_epoch} with best dice {best_dice:.4f}")
# else:
#     logging.info(f"Training from scratch or checkpoint not found. Starting from epoch {start_epoch}")

# # Training loop
# for epoch in range(start_epoch, args.epochs):
#     net.train()
#     running_loss = 0.0

#     for batch in tqdm(dataloader, ncols=80):
#         images, labels = batch['image'].cuda(), batch['label'].cuda()
#         outputs, _ = net(images)

#         loss_ce = ce_loss(outputs, labels)
#         loss_dice = dice_loss(outputs, labels)
#         loss = loss_ce + loss_dice

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     avg_loss = running_loss / len(dataloader)
#     logging.info(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

#     # Evaluation
#     if (epoch + 1) % 2 == 0:
#         net.eval()
#         dice_score = test_3d_patch.var_all_case_LA(
#             net, num_classes=2, patch_size=(112, 112, 80), stride_xy=18, stride_z=4
#         )
#         logging.info(f"Validation Dice Score: {dice_score:.4f}")

#         # Save best model
#         best_model_path = os.path.join(args.save_path, "best_model.pth.tar")
#         if dice_score > best_dice:
#             if os.path.exists(best_model_path):
#                 shutil.copy(best_model_path, best_model_path.replace(".pth.tar", "_backup.pth.tar"))
#                 logging.info("Previous best model backed up.")

#             best_dice = dice_score
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': net.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'best_dice': best_dice
#             }, best_model_path)
#             logging.info(f"Saved new best model to {best_model_path}")

#     # Always save last checkpoint
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': net.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'best_dice': best_dice
#     }, checkpoint_path)

# logging.info(f"Training complete. Best Dice Score: {best_dice:.4f}")



import os
import sys
import argparse
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

from networks.ResVNet import ResVNet
from pancreas.Vnet import VNet
from dataloaders.LADataset import LAHeart
from utils.LA_utils import to_cuda
from pancreas.losses import DiceLoss
from utils import test_3d_patch

# ------------------ Argument parser ------------------ #
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/content/drive/MyDrive/0SSL/Dataset/2018_UTAH_MICCAI')
parser.add_argument('--list_path', type=str, default='/content/drive/MyDrive/0SSL/rakibiuict/LA_SSL/Datasets/la/data_split')
parser.add_argument('--exp', type=str, default='Supervised')
parser.add_argument('--model', type=str, default='VNet', choices=['VNet', 'ResVNet'])
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save_path', type=str, default='/content/drive/MyDrive/0SSL/rakibiuict/model/PreTrained')
args = parser.parse_args()

# ------------------ Setup ------------------ #
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

logging.basicConfig(filename=os.path.join(args.save_path, "train_log.txt"),
                    level=logging.INFO, format='%(asctime)s %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(str(args))

# ------------------ Reproducibility ------------------ #
seed = 1337
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# ------------------ Dataset ------------------ #
dataset = LAHeart(args.root_path, args.list_path, split='train')
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

# ------------------ Model ------------------ #
if args.model == 'VNet':
    net = VNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
else:
    net = ResVNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
net = nn.DataParallel(net).cuda()

# ------------------ Loss and Optimizer ------------------ #
ce_loss = nn.CrossEntropyLoss()
dice_loss = DiceLoss(nclass=2)
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# ------------------ Resume Setup ------------------ #
start_epoch = 0  # <<< CHANGE HERE to resume training
best_dice = 0
checkpoint_path = os.path.join(args.save_path, "last_checkpoint.pth")

if start_epoch > 0 and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_dice = checkpoint.get('best_dice', 0)
    logging.info(f"Resumed training from epoch {start_epoch} with best dice {best_dice:.4f}")
else:
    logging.info(f"Training from scratch or checkpoint not found. Starting from epoch {start_epoch}")

# ------------------ Metrics Excel Setup ------------------ #
excel_path = os.path.join(args.save_path, 'metrics.xlsx')
if os.path.exists(excel_path):
    metrics = pd.read_excel(excel_path).to_dict('records')
    logging.info(f"Loaded existing metrics from {excel_path}")
else:
    metrics = []

# ------------------ Training Loop ------------------ #
for epoch in range(start_epoch, args.epochs):
    net.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, ncols=80):
        images, labels = batch['image'].cuda(), batch['label'].cuda()
        outputs, _ = net(images)

        loss_ce = ce_loss(outputs, labels)
        loss_dice = dice_loss(outputs, labels)
        loss = loss_ce + loss_dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    dice_score = None

    # ------------------ Evaluation ------------------ #
    if (epoch + 1) % 2 == 0:
        net.eval()
        dice_score = test_3d_patch.var_all_case_LA(
            net, num_classes=2, patch_size=(112, 112, 80), stride_xy=18, stride_z=4
        )
        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}, Dice: {dice_score:.4f}")

        # Save best model
        best_model_path = os.path.join(args.save_path, "best_model.pth")
        if dice_score > best_dice:
            if os.path.exists(best_model_path):
                shutil.copy(best_model_path, best_model_path.replace(".pth", "_backup.pth"))
                logging.info("Previous best model backed up.")
            best_dice = dice_score
            torch.save(net.state_dict(), best_model_path)
            logging.info(f"Saved new best model to {best_model_path}")
    else:
        dice_score = -1  # no evaluation

    # ------------------ Save Metrics ------------------ #
    metrics.append({
        'epoch': epoch + 1,
        'loss': avg_loss,
        'dice': round(dice_score, 4) if dice_score != -1 else None
    })

    # Save checkpoint for resume
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice
    }, checkpoint_path)

# ------------------ Final Excel Save ------------------ #
df_metrics = pd.DataFrame(metrics)
df_metrics.drop_duplicates(subset='epoch', keep='last', inplace=True)
df_metrics.sort_values(by='epoch', inplace=True)
df_metrics.to_excel(excel_path, index=False)
logging.info(f"Saved updated metrics to {excel_path}")
logging.info(f"Training complete. Best Dice Score: {best_dice:.4f}")
