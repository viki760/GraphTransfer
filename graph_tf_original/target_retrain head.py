# OTCE
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


source_id = 1
task_names = pd.read_csv("./data/new_target_task_names.csv",index_col=0)

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
domain = "real"
task_index = 0
transfer_b = {}

for source_model_index in range(3,4):
    items = {}
    target_index = 0
    for target_name in task_names.columns:
        target_index = target_index + 1
        print(len(task_names[target_name].values))

        # 在每次运行时生成唯一的日志目录名称
        log_dir = "logs/target_last_head_new_list_{}_".format(source_model_index) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=log_dir)


        print(task_names[target_name].values)
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        model.load_state_dict(torch.load("./save/source_{}_resnet_50_without_pretrained.pth".format(source_model_index)))
        # /home/xiangyuchen/DomainNet_full/0907_tf_estimate/save/source_1_from_scratch_resnet18.pth
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        target_train_dataset = DomainNetDataset('../DomainNet/', transform = composed_transform, is_training=True, domain=domain ,category_list=task_names[target_name].values, num_sample=100)
        target_train_loader = DataLoader(target_train_dataset, batch_size = 10, shuffle=True,  num_workers = 1, drop_last = True)

        target_test_dataset = DomainNetDataset('../DomainNet/', transform = composed_transform, is_training=False, domain=domain ,category_list=task_names[target_name].values, num_sample=100)
        target_test_loader = DataLoader(target_test_dataset, batch_size = 10, shuffle=True,  num_workers = 1, drop_last = True)

        # transfer_b[task_index] = {}

        # 创建早停计数器对象
        early_stop_counter = EarlyStopCounter(threshold=10)

        for epoch in range(100):
            i_index = 0 
            train_epoch_loss = []
            train_epoch_acc = []
            test_epoch_acc = []
            test_epoch_loss = []
            batch_index = 0
            for image, label in target_train_loader:
                batch_index = batch_index + 1
                # print(batch_index)
                image = image.to(device)
                label = label.to(device)
                i_index = i_index + 1
                model.train()
                train_acc = 0.0
                optimizer.zero_grad()  # Reset gradients to zero
                # fine-tuning
                outputs = model(image)
                loss = criterion(outputs, label)
                _, preds = torch.max(outputs, 1)
                train_acc += torch.sum(preds == label.data)
                train_acc /= len(image)
                train_epoch_loss.append(loss.item()* image.size(0))
                train_epoch_acc.append(train_acc.item())
                loss.backward()
                optimizer.step()

            for image, label in target_test_loader:
                batch_index = batch_index + 1
                # print(batch_index)
                image = image.to(device)
                label = label.to(device)
                i_index = i_index + 1
                model.eval()
                test_acc = 0.0
                # fine-tuning
                outputs = model(image)
                test_loss = criterion(outputs, label)
                _, preds = torch.max(outputs, 1)
                test_acc += torch.sum(preds == label.data)
                test_acc /= len(image)
                test_epoch_loss.append(test_loss.item()* image.size(0))
                test_epoch_acc.append(test_acc.item())
            print("epoch:{}, test_loss:".format(epoch),np.mean(test_epoch_loss),"test_acc:",np.mean(test_epoch_acc))
            writer.add_scalar('test_loss', np.mean(test_epoch_loss),epoch)
            writer.add_scalar('test_acc', np.mean(test_epoch_acc),epoch)
            # 判断是否停止训练
            if early_stop_counter.is_stop_training(np.mean(test_epoch_acc)):
                # 停止训练的操作
                print("Training stopped due to early stopping.")
                break
            else:
                print("continue")
        # task_result = {target_name: np.mean(epoch_acc)}
        # transfer_b.setdefault(task_index, {}).update(task_result)
        items[target_name] = np.mean(test_epoch_acc)
        transfer_b[source_model_index] = items
        with open('source_model_{}_from_scrach_new_lit.json'.format(source_model_index), 'w') as f:
            json.dump(transfer_b, f)
        print(transfer_b,source_model_index, target_name)
    

with open('source_model_{}_from_scrach_all_new_list.json'.format(source_model_index), 'w') as f:
    json.dump(transfer_b, f)