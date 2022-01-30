import numpy as np
import pandas as pd
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return torch.Tensor(data), label

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        data = self.data[index]
        return torch.Tensor(data)

    def __len__(self):
        return len(self.data)


class GetData:
    def __init__(self, train_path, test_path):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

    @staticmethod
    def reduce_mem_usage(df):

        # 处理前 数据集总内存计算
        start_mem = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

        # 遍历特征列
        for col in df.columns:
            # 当前特征类型
            col_type = df[col].dtype
            # 处理 numeric 型数据
            if col_type != object:
                c_min = df[col].min()  # 最小值
                c_max = df[col].max()  # 最大值
                # int 型数据 精度转换
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df[col] = df[col].astype(np.int64)
                # float 型数据 精度转换
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            # 处理 object 型数据
            else:
                df[col] = df[col].astype("category")  # object 转 category

        # 处理后 数据集总内存计算
        end_mem = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

        return df

    def data_processing(self):
        train_list = []
        for items in self.train.values:
            train_list.append(
                [items[0]] + [float(i) for i in items[1].split(",")] + [items[2]]
            )
        train = pd.DataFrame(np.array(train_list))
        train.columns = (
            ["id"] + ["s_" + str(i) for i in range(len(train_list[0]) - 2)] + ["label"]
        )  # 特征分离
        self.train = self.reduce_mem_usage(train)  # 精度量化

        test_list = []
        for items in self.test.values:
            test_list.append([items[0]] + [float(i) for i in items[1].split(",")])
        test = pd.DataFrame(np.array(test_list))
        test.columns = ["id"] + [
            "s_" + str(i) for i in range(len(test_list[0]) - 1)
        ]  # 特征分离
        self.test = self.reduce_mem_usage(test)  # 精度量化

    def convert_data(self):
        # 查看训练集, 分离标签与样本, 去除 id
        self.train_label = self.train["label"]
        self.train_data = self.train.drop(["id", "label"], axis=1)

        # 查看测试集, 去除 id
        self.test_data = self.test.drop(["id"], axis=1)

    def get_train_data(self):
        self.data_processing()
        self.convert_data()
        train_data = self.train_data.values[:, np.newaxis, :]
        train_label = self.train_label.values
        return train_data, train_label

    def get_test_data(self):
        self.data_processing()
        self.convert_data()
        test_data = self.test_data.values[:, np.newaxis, :]
        return test_data
