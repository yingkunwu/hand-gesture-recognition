import torch, os
from tqdm import tqdm

from libs.load import load_data
from libs.options import TrainOptions
from model.CPM import ConvolutionalPoseMachine
from libs.loss import mse_loss, ce_loss


class Train:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ConvolutionalPoseMachine(self.args.num_masks, self.args.num_heatmaps)


    def train(self):
        train_set, valid_set, train_dataloader, val_dataloader = load_data(self.args.data_folder, self.args.img_size, 
                                                                        self.args.sigma, self.args.batch_size, "train")
        print("The number of data in train set: ", train_set.__len__())
        print("The number of data in valid set: ", valid_set.__len__())

        self.model = self.model.to(self.device)

        # define loss function and optimizer
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)

        for i in range(self.args.epochs):
            train_loss, val_loss = 0, 0
            train_acc, val_acc = 0, 0

            self.model.train()
            for i, (images, keypoints, limbmasks, labels) in enumerate(tqdm(train_dataloader)):
                images = images.to(self.device)
                keypoints = keypoints.to(self.device)
                limbmasks = limbmasks.to(self.device)

                g6_targ = limbmasks[:, :1, ...]
                g1_targ = limbmasks[:, 1:, ...]
                g6_targ = torch.cat([g6_targ] * 3, dim=1)
                g1_targ = torch.cat([g1_targ] * 3, dim=1)
                kp_targ = torch.cat([keypoints] * 3, dim=1)  

                g6_pred, g1_pred, kp_pred = self.model(images)

                g1_loss = ce_loss(g1_pred, g1_targ)
                g6_loss = ce_loss(g6_pred, g6_targ)
                kp_loss = mse_loss(kp_pred, kp_targ)

                loss = g1_loss + g6_loss + kp_loss

                #print("g1_loss: {}, g6_loss: {}, kp_loss: {}".format(g1_loss.item(), g6_loss.item(), kp_loss.item()))

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                #prediction = torch.argmax(outputs.detach(), dim=1)
                #train_acc += torch.mean(torch.eq(prediction, labels).type(torch.float32)).item()
                
            print("Epoch: {}, train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}"
                    .format(i + 1, train_loss, train_acc, val_loss, val_acc))
            torch.save(self.model.state_dict(), os.path.join("weights", self.args.model_name))


if __name__ == "__main__":
    parser = TrainOptions()
    args = parser.parse()
    print(args)

    t = Train(args)
    t.train()
