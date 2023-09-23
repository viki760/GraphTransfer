import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from PIL import Image
import os
import numpy as np
import pickle
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt


import numpy as np
import ot
import geomloss
import torch
import math
from PIL import Image
import os
import numpy as np
import pickle
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import json

# task_embedding[name] = {"feature":feature,"label":label}
import random

from tensorboardX import SummaryWriter

# 在指定目录下创建一个新的Tensorboard实例
from torch.utils.tensorboard import SummaryWriter
import datetime

import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
import numpy as np
np.random.seed(0)
import random
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class EarlyStopCounter:
    def __init__(self, threshold):
        self.threshold = threshold
        self.counter = 0
        self.best_acc = 0.0

    def is_stop_training(self, current_acc):
        if current_acc > self.best_acc:
            self.best_acc = current_acc
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.threshold:
            return True
        else:
            return False
        
# 检查是否可用GPU
if torch.cuda.is_available():
    # 将模型移动到GPU上
    device = torch.device("cuda")


# 种类映射成index
with open("cate_map.json","r") as f:
    cate_map = json.loads(f.read())

import pandas as pd

target_task_names = pd.read_csv("./data/new_target_task_names.csv",index_col=0)
# task_names = pd.read_csv("target_task_names.csv",index_col=0)


class DomainNetDataset(Dataset):
    
    def __init__(self, data_root, transform=None, is_training=True, domain=None, category_list = None, num_sample = None, require_file_path=False):

        self.data_root = data_root
        self.transform = transform
        self.domain = domain
        self.require_file_path = require_file_path

        if domain == 'clipart':
            if is_training:
                file_txt = open(os.path.join(self.data_root,'clipart_train.txt'),'r')
            else:
                file_txt = open(os.path.join(self.data_root,'clipart_test.txt'),'r')

        elif domain == 'infograph':
            if is_training:
                file_txt = open(os.path.join(self.data_root,'infograph_train.txt'),'r')
            else:
                file_txt = open(os.path.join(self.data_root,'infograph_test.txt'),'r')

        elif domain == 'painting':
            if is_training:
                file_txt = open(os.path.join(self.data_root,'painting_train.txt'),'r')
            else:
                file_txt = open(os.path.join(self.data_root,'painting_test.txt'),'r')
        
        elif domain == 'quickdraw':
            if is_training:
                file_txt = open(os.path.join(self.data_root,'quickdraw_train.txt'),'r')
            else:
                file_txt = open(os.path.join(self.data_root,'quickdraw_test.txt'),'r')

        elif domain == 'real':
            if is_training:
                file_txt = open(os.path.join(self.data_root,'real_train.txt'),'r')
            else:
                file_txt = open(os.path.join(self.data_root,'real_test.txt'),'r')

        elif domain == 'sketch':
            if is_training:
                file_txt = open(os.path.join(self.data_root,'sketch_train.txt'),'r')
            else:
                file_txt = open(os.path.join(self.data_root,'sketch_test.txt'),'r')


        self.file_list = []
        for line in file_txt.readlines():
            file, label = line.strip().split(' ')
            self.file_list.append({'file':file, 'label':int(label)})


        rdm = np.random.RandomState(20200816)
        self.selected_file_list = []
        if len(category_list):            
            for new_label, category in enumerate(category_list):
                cur_file_list = [{'file':each['file'], 'label': new_label, 'old_label': each['label']} for each in self.file_list if each['label'] == cate_map[category]]
                
                if num_sample and len(cur_file_list) > num_sample:
                    idx_list = [i for i in range (0,len(cur_file_list))]
                    sampled_id = rdm.choice(idx_list,num_sample,replace=False).tolist()

                    self.selected_file_list += [cur_file_list[i] for i in sampled_id]
                else:
                    self.selected_file_list += cur_file_list

        else:
            self.selected_file_list = self.file_list


    def _save_selected_data(self,file_path):

        f = open(file_path,'w')
        json.dump(self.selected_file_list, f)
        f.close()


    def __len__(self):

        return len(self.selected_file_list)


    def __getitem__(self,idx,require_file_path=False):

        img = Image.open(os.path.join(self.data_root,self.selected_file_list[idx]['file']))
        if self.transform:
            img = self.transform(img)
        label = self.selected_file_list[idx]['label']
        if self.require_file_path:
            return img, label, self.selected_file_list[idx]['file']
        else:
            return img, label


composed_transform = transforms.Compose(
                        [transforms.Resize (224),
                        transforms.CenterCrop((224, 224)),
                        transforms.ToTensor()])


domain = 'real'
transfer_b = {}
learning_rate = 0.0001

task_index = 1
log_dir = "logs/target_tfs_{}".format(task_index)
tfs_acc = {}

for target_task_name in target_task_names.columns:
    task_index = task_index + 1
    # import pdb; pdb.set_trace()
    # source_task_name = "task1"
    print(len(target_task_names[target_task_name].values))
    # 在每次运行时生成唯一的日志目录名称
    writer = SummaryWriter(log_dir=log_dir)

    # 全训练
    model = models.resnet50(pretrained=False)
    # model.load_state_dict(torch.load("/home/xiangyuchen/DomainNet_full/resnet50-19c8e357.pth"))
    for param in model.parameters():
        param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)


    source_train_dataset = DomainNetDataset('../DomainNet/', transform=composed_transform, is_training=True, domain=domain ,category_list=source_task_names[source_task_name].values, num_sample=150)
    source_train_loader = DataLoader(source_train_dataset, batch_size = 10, shuffle=True,  num_workers = 1, drop_last = True)

    source_test_dataset = DomainNetDataset('../DomainNet/', transform=composed_transform, is_training=False, domain=domain ,category_list=source_task_names[source_task_name].values, num_sample=150)
    source_test_loader = DataLoader(source_test_dataset, batch_size = 10, shuffle=True,  num_workers = 1, drop_last = True)

    i_index = 0    
    # transfer_b[source_task_name] = {}

    early_stop_counter = EarlyStopCounter(threshold=20)

    for epoch in range(1000):
        i_index = 0 
        train_acc = []
        epoch_acc = []
        epoch_loss = []
        batch_index = 0
        model.train()
        for image, label in source_train_loader:
            batch_index = batch_index + 1
            # print(batch_index)
            image = image.to(device)
            label = label.to(device)
            i_index = i_index + 1
            optimizer.zero_grad()  # Reset gradients to zero
            train_acc = 0.0
            # fine-tuning
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == label.data)
            train_acc /= len(image)
            # print(train_acc)
            train_acc.append(train_acc.item())
        print("train_acc:",np.mean(train_acc))

        model.eval()
        for image, label in source_test_loader:
            batch_index = batch_index + 1
            # print(batch_index)
            image = image.to(device)
            label = label.to(device)
            i_index = i_index + 1
            test_acc = 0.0
            # fine-tuning
            outputs = model(image)
            loss = criterion(outputs, label)
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == label.data)
            test_acc /= len(image)
            epoch_loss.append(loss.item()* image.size(0))
            epoch_acc.append(test_acc.item())

        print("epoch:{}, test_loss:".format(epoch), np.mean(epoch_loss), "test_acc:",np.mean(epoch_acc))
        writer.add_scalar('test_loss', np.mean(epoch_loss),epoch)
        writer.add_scalar('target_{}_test_tfs'.format(task_index), np.mean(epoch_acc),epoch)

        # 判断是否停止训练
        if early_stop_counter.is_stop_training(np.mean(epoch_acc)):
            # 停止训练的操作
            print("Training stopped due to early stopping.")
            tfs_acc[target_task_name] = np.mean(epoch_acc)
            with open(('target_tfs.json'), 'w') as f:
                json.dump(tfs_acc, f)
            break
        else:
            print("continue")
        

        state_dict = model.state_dict()
        torch.save(state_dict, './save/target_{}_tfs.pth'.format(task_index))
        print("target_task:", task_index)
    
    with open(('target_tfs.json'), 'w') as f:
        json.dump(tfs_acc, f)


        # for target_name in task_names.columns:
        #     print(task_names[target_name].values)

        #     model = models.resnet50(pretrained=False)
        #     model.load_state_dict(torch.load("./save/source_{i}.pth"))
        #     for param in model.parameters():
        #         param.requires_grad = False
        #     num_ftrs = model.fc.in_features
        #     model.fc = nn.Linear(num_ftrs, 10)
        #     model = model.to(device)

        #     target_dataset = DomainNetDataset('./DomainNet/', transform=composed_transform, is_training=True, domain=domain ,category_list=task_names[target_name].values, num_sample=100)
        #     target_loader = DataLoader(target_dataset, batch_size = 10, shuffle=True,  num_workers = 1, drop_last = True)
        #     model.eval()
        #     val_loss = 0.0
        #     val_acc = 0.0
        #     for inputs, labels in target_loader:
        #         with torch.no_grad():
        #             inputs = inputs.to(device)
        #             labels = labels.to(device)
        #             outputs = model(inputs)
        #             loss = criterion(outputs, labels)
        #             # import pdb; pdb.set_trace()
        #             val_loss += loss.item() * image.size(0)
        #             _, preds = torch.max(outputs, 1)
        #             val_acc += torch.sum(preds == labels.data)
        #             # print("test_loss:",val_loss)
        #     val_loss /= len(target_loader.dataset)
        #     val_acc /= len(target_loader.dataset)

        #     print('test_loss: {:.4f}, test_acc: {:.4f}'.format(val_loss, val_acc))
        #     transfer_b[source_task_name][target_name] = val_acc.item()
        #     with open('0509_transfer_b2.json', 'w') as f:
        #         json.dump(transfer_b, f)

