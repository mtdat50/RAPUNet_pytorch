import torch
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from ModelArchitecture.RAPUNet import RAPUNet
from Utils.DiceLoss import dice_loss
from Utils.CustomDataset import FolderDataset, glob_paths


def test(model, test_loader, loss_fn, device):
    model.eval()
    model.to(device)

    total_loss = 0.0
    sample_count = 0
    for batch_count, (batch_input, batch_gt) in enumerate(test_loader):
        batch_input = batch_input.to(device)
        batch_gt = batch_gt.to(device)

        with torch.no_grad():
            batch_pred = model(batch_input)
            batch_loss = loss_fn(batch_pred, batch_gt)

        total_loss += batch_loss.item() * batch_input.shape[0]
        sample_count += batch_input.shape[0]
        print(f"Testing ({batch_count + 1}/{len(test_loader)}): Loss {total_loss / sample_count}" + ' '*50, end='\r')
    print('\n')

    return total_loss / sample_count


aug_test = albu.Compose([
    albu.Resize(384, 384, interpolation=cv2.INTER_LANCZOS4, always_apply=True),
    ToTensorV2()  # Ensure output is a PyTorch tensor
])

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    batch_size = 16
    
    model_path = "/kaggle/working/bestmodel_epoch93_loss0.12060173153877259.pt"
    model = RAPUNet(in_channels=3, out_classes=1)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    loss_fn = dice_loss

    
    datasets = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]
    for dataset in datasets:
        print(f"{dataset}:")
        test_path = f"/kaggle/input/polyp-dataset/data/TestDataset/{dataset}/"
    
        X, Y = glob_paths(test_path)
        test_set = FolderDataset(X, Y, transform=aug_test)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)
    
        test_loss = test(model, test_loader, loss_fn, device)
        with open(f"/kaggle/working/test_log.txt", "a") as f:
            f.write(f"{dataset}: Loss {test_loss}\n")
            f.write(f"{dataset}: mDice {1 - test_loss}\n\n")