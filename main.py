import torch, time
from tqdm import tqdm
import numpy as np
import cv2

from load import load_data
from model.CPM import ConvolutionalPoseMachine
from torchsummary import summary


def predict(weight_path, data_dir, batch_size):
    model = ConvolutionalPoseMachine(21)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

    train_set, valid_set, train_dataloader, val_dataloader = load_data(data_dir, batch_size)
    print("The number of data in train set: ", train_set.__len__())
    print("The number of data in valid set: ", valid_set.__len__())
    
    start_time = time.time()

    # --------------------------
    # Testing Stage
    # --------------------------
    model.eval()
    with torch.no_grad():
        correct_test = 0.0
        for i, (images, landmarks, labels) in enumerate(tqdm(val_dataloader)):
            heatmap1, heatmap2, heatmap3, heatmap4 = model(images)

            skeletons = heatmap4[0]
            skeletons = np.array(skeletons).transpose(1, 2, 0)
            images = images * 0.5 + 0.5
            img = images[0]
            img = np.array(img).transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for i in range(22):
                skeleton = skeletons[:, :, i]
                print(skeleton)
                skeleton = cv2.resize(skeleton, (368, 368))
                skeleton = cv2.normalize(skeleton, skeleton, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                skeleton = cv2.applyColorMap(skeleton, cv2.COLORMAP_JET)
                
                display = img * 0.8 + skeleton * 0.2
                print("predict")
                cv2.imshow("img", display)
                cv2.waitKey(0)

    end_time = time.time()

    print('%2.2f sec(s) Test Acc: %3.6f' % (end_time - start_time, correct_test / valid_set.__len__()))


def train(data_dir, batch_size, learning_rate, momentum, weight_decay, epochs):
    train_set, valid_set, train_dataloader, val_dataloader = load_data(data_dir, batch_size)
    print("The number of data in train set: ", train_set.__len__())
    print("The number of data in valid set: ", valid_set.__len__())

    model = ConvolutionalPoseMachine(21)

    # define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    heat_weight = 46 * 46 * 22 / 1.0

    for i in range(epochs):
        train_loss, val_loss = 0, 0
        train_acc, val_acc = 0, 0

        model.train()
        for j, (images, landmarks, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            heatmap1, heatmap2, heatmap3, heatmap4 = model(images)
        
            loss1 = criterion(heatmap1, landmarks) * heat_weight
            loss2 = criterion(heatmap2, landmarks) * heat_weight
            loss3 = criterion(heatmap3, landmarks) * heat_weight
            loss4 = criterion(heatmap4, landmarks) * heat_weight

            loss = loss1 + loss2 + loss3 + loss4

            print("loss1: {}, loss2: {}, loss3: {}, loss4: {}".format(loss1.item(), loss2.item(), loss3.item(), loss4.item()))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            #prediction = torch.argmax(outputs.detach(), dim=1)
            #train_acc += torch.mean(torch.eq(prediction, labels).type(torch.float32)).item()

        model.eval()
        for j, (images, landmarks, labels) in enumerate(tqdm(val_dataloader)):
            with torch.no_grad():

                heatmap1, heatmap2, heatmap3, heatmap4 = model(images)
            
                loss1 = criterion(heatmap1, landmarks) * heat_weight
                loss2 = criterion(heatmap2, landmarks) * heat_weight
                loss3 = criterion(heatmap3, landmarks) * heat_weight
                loss4 = criterion(heatmap4, landmarks) * heat_weight

                loss = loss1 + loss2 + loss3 + loss4
                val_loss += loss.item()
                #prediction = torch.argmax(outputs.detach(), dim=1)
                #val_acc += torch.mean(torch.eq(prediction, labels).type(torch.float32)).item()
            
        train_loss /= train_dataloader.__len__()
        train_acc /= train_dataloader.__len__()
        val_loss /= val_dataloader.__len__()
        val_acc /= val_dataloader.__len__()
        print("Epoch: {}, train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}"
                .format(i + 1, train_loss, train_acc, val_loss, val_acc))
        torch.save(model.state_dict(), "weights/CPM.model")


if __name__ == "__main__":
    data_dir = "hagrid/hagrid_resize"
    weight_path = "weights/CPM.model"
    batch_size = 16
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 0.0
    epochs = 10

    #model = ConvolutionalPoseMachine(21)
    #summary(model, (3, 368, 368))

    #train(data_dir, batch_size, learning_rate, momentum, weight_decay, epochs)
    predict(weight_path, data_dir, 1)
