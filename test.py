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
    for batch_count, (batch_input, batch_gt) in enumerate(test_loader):
        batch_input = batch_input.to(device)
        batch_gt = batch_gt.to(device)

        with torch.no_grad():
            batch_pred = model(batch_input)
            batch_loss = loss_fn(batch_pred, batch_gt)

        total_loss += batch_loss.item()
        print(f"Validating ({batch_count + 1}/{len(test_loader)}): Loss {total_loss / (batch_count + 1)}" + ' '*50, end='\r')
    print('\n')

    return total_loss / len(test_loader)


aug_test = albu.Compose([
    albu.Resize(384, 384, interpolation=cv2.INTER_LANCZOS4, always_apply=True),
    ToTensorV2()  # Ensure output is a PyTorch tensor
])

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    batch_size = 16
    starting_kernels = 16

    datasets = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]
    test_path = f"../data/TestDataset/{datasets[3]}/"

    X, Y = glob_paths(test_path)
    test_set = FolderDataset(X, Y, transform=aug_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)


    model_path = "/kaggle/working/.........."
    model = RAPUNet(in_channels=3, out_classes=1, starting_kernels=starting_kernels)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    loss_fn = dice_loss

    test_loss = test(model, test_loader, loss_fn, device)
    with open("/kaggle/working/test_log.txt", "a") as f:
        f.write(f"Validating: Loss {test_loss / len(test_loader)}\n\n")