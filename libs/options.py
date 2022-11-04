import argparse

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--data_folder", type=str, default=None, help="path to train dataset")
        self.parser.add_argument("--model_name", type=str, default=None, help="new model name")
        self.parser.add_argument("--epochs", type=int, default=None, help="number of epochs")
        self.parser.add_argument("--lr", type=float, default=None, help="learning rate")
        self.parser.add_argument("--batch_size", type=int, default=None, help="size of batches")
        self.parser.add_argument("--img_size", type=int, default=None, help="size of each image dimension")
        self.parser.add_argument("--sigma", type=int, default=None, help="standard deviation of heatmap gaussian distribution")
        self.parser.add_argument("--num_masks", type=int, default=7, help="number of masks of limbs")
        self.parser.add_argument("--num_heatmaps", type=int, default=22, help="number of landmarks; should be K+1")

    def parse(self):
        p = self.parser.parse_args()
        temp = []
        if p.data_folder is None:
            temp.append("--data_folder")
        if p.model_name is None:
            temp.append("--model_name")
        if p.epochs is None:
            temp.append("--epochs")
        if p.lr is None:
            temp.append("--lr")
        if p.batch_size is None:
            temp.append("--batch_size")
        if p.img_size is None:
            temp.append("--img_size")
        if p.sigma is None:
            temp.append("--sigma")
        if len(temp) > 0:
            raise argparse.ArgumentTypeError("no argument specified in {}".format(temp))
        return p

class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--data_folder", type=str, default=None, help="path to dataset")
        self.parser.add_argument("--model_name", type=str, default=None, help="path to pretrained weights file")
        self.parser.add_argument("--batch_size", type=int, default=None, help="size of the batches")
        self.parser.add_argument("--img_size", type=int, default=None, help="size of each image dimension")
        self.parser.add_argument("--sigma", type=int, default=None, help="standard deviation of heatmap gaussian distribution")
        self.parser.add_argument("--num_masks", type=int, default=7, help="number of masks of limbs")
        self.parser.add_argument("--num_heatmaps", type=int, default=22, help="number of landmarks; should be K+1")

    def parse(self):
        p = self.parser.parse_args()
        temp = []
        if p.data_folder is None:
            temp.append("--data_folder")
        if p.model_name is None:
            temp.append("--model_name")
        if p.batch_size is None:
            temp.append("--batch_size")
        if p.img_size is None:
            temp.append("--img_size")
        if p.sigma is None:
            temp.append("--sigma")
        if len(temp) > 0:
            raise argparse.ArgumentTypeError("no argument specified in {}".format(temp))
        return p
