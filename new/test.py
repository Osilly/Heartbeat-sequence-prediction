import os
from torch.utils.data import DataLoader
import numpy as np
from glob import glob
from network import *
from dataset import *
import random
from pandas import DataFrame


class Test:
    def __init__(
        self,
        train_path="data/train.csv",
        test_path="data/testA.csv",
        result_path="result",
        num_classes=4,
        batch_size=200,
        block_config=[6, 12, 64, 48],
        device="cuda:0",
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.result_path = result_path
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.block_config = block_config
        self.device = device

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
        test_data = getdata.get_test_data()
        test_iter = TestDataset(test_data)
        self.test_iter = DataLoader(test_iter, batch_size=self.batch_size, shuffle=True)

    def build_model(self):
        self.net = DenseNet(
            growth_rate=32,
            block_config=self.block_config,
            num_init_features=64,
            bn_size=4,
            drop_rate=0,
            num_classes=self.num_classes,
        ).to(device=self.device)

    def load_model(self, path, step):
        params = torch.load(os.path.join(path, "model_params_%07d.pt" % step))
        self.net.load_state_dict(params["net"])
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

    def test(self):
        model_list = glob(os.path.join(self.result_path, "model", "*.pt"))
        if len(model_list) != 0:
            model_list.sort()
            start_step = int(model_list[-1].split("_")[-1].split(".")[0])
            self.load_model(os.path.join(self.result_path, "model"), start_step)
            print("load success!")

        self.net.eval()
        result = []
        for data in self.test_iter:
            data = data.to(self.device)
            with torch.no_grad():
                out = self.net(data)
            pre = F.softmax(out, 1)
            pre = pre.to("cpu")
            result.append(pre)
        result = torch.stack(result, 0)  # 按照轴0将list转换为tensor
        # 进行数据的后处理，准备提交数据(设置阈值)
        result = result.numpy()
        result = result.reshape((20000, 4))
        thr = [0.8, 0.45, 0.8, 0.8]
        for x in result:
            for i in [1, 2, 3, 0]:
                if x[i] > thr[i]:
                    x[0:i] = 0
                    x[i + 1 : 4] = 0
                    x[i] = 1

        id = np.arange(100000, 120000)
        df = DataFrame(result, columns=["label_0", "label_1", "label_2", "label_3"])
        df.insert(loc=0, column="id", value=id, allow_duplicates=False)
        df.to_csv("submit.csv", index_label="id", index=False)
        print(df.head())
