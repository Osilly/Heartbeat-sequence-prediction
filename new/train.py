import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from glob import glob
from network import *
from dataset import *
import random
import time


class Train:
    def __init__(
        self,
        train_path="data/train.csv",
        test_path="data/testA.csv",
        result_path="result",
        num_classes=4,
        growth_rate=32,
        block_config=[6, 12, 64, 48],
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        num_epochs=50,
        batch_size=200,
        lr=1e-1,
        weight_decay=1e-4,
        device="cuda:0",
        resume=False,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.result_path = result_path
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.resume = resume

        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []
        self.start_epoch = 1

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def dataload(self):
        getdata = GetData(train_path=self.train_path, test_path=self.test_path)
        train_data, train_label = getdata.get_train_data()
        train_size = int(len(train_data) * 0.85)
        valid_size = len(train_data) - train_size
        train_iter, valid_iter = torch.utils.data.random_split(
            TrainDataset(train_data, train_label), [train_size, valid_size]
        )
        self.train_iter = DataLoader(
            train_iter, batch_size=self.batch_size, shuffle=True
        )
        self.valid_iter = DataLoader(
            valid_iter, batch_size=self.batch_size, shuffle=False
        )

    def build_model(self):
        self.net = DenseNet(
            growth_rate=self.growth_rate,
            block_config=self.block_config,
            num_init_features=self.num_init_features,
            bn_size=self.bn_size,
            drop_rate=self.drop_rate,
            num_classes=self.num_classes,
        ).to(device=self.device)

    def define_loss(self):
        # 评估函数
        # class MyLoss(nn.Module):
        #     def __init__(self):
        #         super().__init__()

        #     def forward(self, out, label):
        #         label = nn.functional.one_hot(label, num_classes=4)
        #         loss = sum(sum(abs(out - label)))
        #         return loss

        # self.loss = MyLoss().to(device=self.device)
        self.loss = nn.CrossEntropyLoss().to(device=self.device)

    def define_optim(self):
        self.optim = optim.SGD(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.9)

    def save_model(self, path, step):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        params = {}
        params["net"] = self.net.state_dict()
        params["optim"] = self.optim.state_dict()
        params["train_loss"] = self.train_loss
        params["valid_loss"] = self.valid_loss
        params["train_acc"] = self.train_acc
        params["valid_acc"] = self.valid_acc
        params["start_epoch"] = self.start_epoch
        torch.save(params, os.path.join(path, "model_params_%07d.pt" % step))

    def load_model(self, path, step):
        params = torch.load(os.path.join(path, "model_params_%07d.pt" % step))
        self.net.load_state_dict(params["net"])
        self.optim.load_state_dict(params["optim"])
        self.train_loss = params["train_loss"]
        self.valid_loss = params["valid_loss"]
        self.train_acc = params["train_acc"]
        self.valid_acc = params["valid_acc"]
        self.start_epoch = params["start_epoch"]

    @staticmethod
    def get_acc(out, label):
        total = out.shape[0]
        _, pred_label = out.max(1)
        num_correct = (pred_label == label).sum().item()
        return num_correct / total

    def train(self):
        if self.resume:
            model_list = glob(os.path.join(self.result_path, "model", "*.pt"))
            if len(model_list) != 0:
                model_list.sort()
                start_step = int(model_list[-1].split("_")[-1].split(".")[0])
                self.load_model(os.path.join(self.result_path, "model"), start_step)
                print("load success!")
        print("training started!")
        for epoch in range(self.start_epoch, 1 + self.num_epochs):
            self.net.train()
            train_loss = 0
            train_acc = 0
            valid_loss = 0
            valid_acc = 0
            # train
            start = time.time()
            for data, label in self.train_iter:
                data, label = (
                    data.to(device=self.device),
                    label.to(dtype=torch.long, device=self.device),
                )
                out = self.net(data)
                loss = self.loss(out, label)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                train_loss += loss.item()
                train_acc += self.get_acc(out, label)
            self.scheduler.step()
            train_loss = train_loss / len(self.train_iter)
            train_acc = train_acc / len(self.train_iter)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            # valid
            self.net.eval()
            for data, label in self.valid_iter:
                data, label = (
                    data.to(device=self.device),
                    label.to(dtype=torch.long, device=self.device),
                )
                out = self.net(data)
                loss = self.loss(out, label)
                valid_loss += loss.item()
                valid_acc += self.get_acc(out, label)
            valid_loss = valid_loss / len(self.valid_iter)
            valid_acc = valid_acc / len(self.valid_iter)
            self.valid_loss.append(valid_loss)
            self.valid_acc.append(valid_acc)

            end = time.time()
            print(
                "[%fs] Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (
                    end - start,
                    epoch,
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                )
            )

            if epoch % 50 == 0:
                self.start_epoch = epoch
                self.save_model(os.path.join(self.result_path, "model"), epoch)
