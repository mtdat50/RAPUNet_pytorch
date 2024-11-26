import cv2
import torch
import torch.optim as optim
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ModelArchitecture.RAPUNet import RAPUNet
from Utils.DiceLoss import dice_loss
from Utils.CustomDataset import FolderDataset, glob_paths


def pass_epoch(model, train_loader, val_loader, optimizer, weight_decay_scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch_count, (batch_input, batch_gt) in enumerate(train_loader):
        # weight decay scheduling
        weight_decay = max(weight_decay_scheduler.get_last_lr()[0], 1e-6)
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = weight_decay

        batch_input = batch_input.to(device)
        batch_gt = batch_gt.to(device)

        batch_pred = model(batch_input)
        batch_loss = loss_fn(batch_pred, batch_gt)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        weight_decay_scheduler.step()

        total_loss += batch_loss.item()
        print(f"Training ({batch_count + 1}/{len(train_loader)}): Loss {total_loss / (batch_count + 1)}| Weight decay: {weight_decay}" + ' '*50, end='\r')
    print('')
    with open("/kaggle/working/log.txt", "a") as f:
        f.write(f"Training: Loss {total_loss / len(train_loader)}| Weight decay: {weight_decay}\n")

    model.eval()
    total_loss = 0.0
    for batch_count, (batch_input, batch_gt) in enumerate(val_loader):
        batch_input = batch_input.to(device)
        batch_gt = batch_gt.to(device)

        with torch.no_grad():
            batch_pred = model(batch_input)
            batch_loss = loss_fn(batch_pred, batch_gt)

        total_loss += batch_loss.item()
        print(f"Validating ({batch_count + 1}/{len(val_loader)}): Loss {total_loss / (batch_count + 1)}" + ' '*50, end='\r')
    print('\n')
    with open("/kaggle/working/log.txt", "a") as f:
        f.write(f"Validating: Loss {total_loss / len(val_loader)}\n\n")
    return total_loss / len(val_loader)


aug_train = albu.Compose([
    albu.Resize(384, 384, interpolation=cv2.INTER_LANCZOS4, always_apply=True),
    albu.HorizontalFlip(),
    albu.VerticalFlip(),
    albu.ColorJitter(brightness=(0.6,1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
    albu.Affine(scale=(0.5,1.5), translate_percent=(-0.125,0.125), rotate=(-180,180), shear=(-22.5,22), always_apply=True),
    ToTensorV2()  # Ensure output is a PyTorch tensor
])


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    seed_value = 58800
    img_size = 384
    batch_size = 16
    epochs = 300
    starting_kernels = 16

    train_path = "../data/TrainDataset/"
    X, Y = glob_paths(train_path)
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1, shuffle= True, random_state = seed_value)

    train_set = FolderDataset(x_train, y_train, transform=aug_train)
    val_set = FolderDataset(x_valid, y_valid, transform=aug_train)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)

    model = RAPUNet(in_channels=3, out_classes=1, starting_kernels=starting_kernels)

    initial_weight_decay = 1e-4
    end_weight_decay = 1e-6
    initial_learning_rate = 1e-4
    end_learning_rate = 1e-6
    decay_steps = 1000

    optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=initial_weight_decay)
    tmp = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=initial_weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, min_lr=end_learning_rate)
    weight_decay_scheduler = optim.lr_scheduler.PolynomialLR(tmp, total_iters=decay_steps, power=0.2)
    loss_fn = dice_loss
    
    min_loss_to_save = 0.2

    model = model.to(device)

    for epoch in range(epochs):
        logstr = f"Epoch {epoch + 1}/{epochs}\nLr: {lr_scheduler.get_last_lr()}\n"
        print(logstr, end='')
        with open("/kaggle/working/log.txt", "a") as f:
            f.write(logstr)

        val_loss = pass_epoch(model, train_loader, val_loader, optimizer, weight_decay_scheduler, loss_fn, device)
        lr_scheduler.step(val_loss)

        if val_loss < min_loss_to_save:
            min_loss_to_save = val_loss
            model_path = f"/kaggle/working/bestmodel_epoch{epoch+1}_loss{val_loss}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Saved model after epoch {epoch + 1} with val_loss: {val_loss}")