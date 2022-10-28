import torch, time, torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from load import load_data
from model.ResNet import ResNet


def predict(model, test_set, test_dataloader):
    start_time = time.time()

    # --------------------------
    # Testing Stage
    # --------------------------
    model.eval()
    with torch.no_grad():
        correct_test = 0.0
        for i, (x, label) in enumerate(tqdm(test_dataloader)):
            val_pred = model(x)

            _, predicted = torch.max(val_pred.cpu().data, 1)
            batch_accuracy = (predicted == label.cpu()).sum().item()
            correct_test += batch_accuracy

    end_time = time.time()

    print('%2.2f sec(s) Test Acc: %3.6f' % (end_time - start_time, correct_test / test_set.__len__()))


def imshow(img):
    img = img * 0.5 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(15, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    data_dir = "hagrid/hagrid_resize"
    batch_size = 1

    train_set, valid_set, train_dataloader, val_dataloader = load_data(data_dir, batch_size)
    print(train_set.__len__())
    print(valid_set.__len__())
    for i, (images, landmarks, labels) in enumerate(tqdm(train_dataloader)):
        print(landmarks)
        print(labels)
        imshow(torchvision.utils.make_grid(images))
        exit(1)
        pass
