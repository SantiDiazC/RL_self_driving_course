# from calendar import EPOCH
import os
from platform import java_ver
from random import sample, shuffle
from tkinter import image_types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import pathlib
from torchvision.datasets import ImageFolder
import torch.utils.data as data
import re
from skimage import io, transform
import matplotlib.pyplot as plt
import cv2
from datetime import datetime


#   def do_train(Net_class, train_loader, test_loader, device, EPOCHS, BATCH_SIZE):


def train(angle_model, trottle_model, angle_optimizer, trottle_optimizer, train_loader, log_interval, device):
    trottle_model.train()
    angle_model.train()
    loss_total = 0
    iter = 0

    for batch_idx, sample_batched in enumerate(train_loader): # confrim train_loader
        x_train = torch.permute(sample_batched['image'], (0, 3, 1, 2)).float()
        x_train = x_train.to(device)
        train_label = sample_batched['label'].to(device)

        angle_optimizer.zero_grad()
        angle_label = train_label[:, 0:1]
        angle_output = angle_model(x_train).to(device)
        angle_loss = criterion(angle_output, angle_label)*100

        trottle_optimizer.zero_grad()
        trottle_label = train_label[:,1:2]
        trottle_output = trottle_model(x_train).to(device)
        trottle_loss = criterion(trottle_output, trottle_label)*100
        
        angle_loss.backward()
        angle_optimizer.step()
        trottle_loss.backward()
        trottle_optimizer.step()
        iter += 1

        if batch_idx % log_interval == 0:
            print("train epoch : {} [{}/{}({:.0f}%)]\ttrain loss : {:.6f}".format(Epoch, batch_idx*Batch_size,
                                                                                  len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                                                                                  angle_loss.item()))
        loss_total += angle_loss.item()
    print('epoch finished', loss_total/iter)


def evaluate(angle_model, trottle_model, test_loader, device, Epoch,path):
    angle_model.eval()
    trottle_model.eval()
    test_loss = 0
    iter =0
    with torch.no_grad():
        a=0
        for batch_idx, sample_batched in enumerate(test_loader):    # confirm train_loader
            x_test = sample_batched['image']
            x_test = torch.permute(x_test, (0, 3, 1, 2)).float()
            x_test = x_test.to(device)
            test_label = sample_batched['label'].to(device)
            angle_label = test_label[:,0:1]
            trottle_label = test_label[:,1:2]


            angle_output = angle_model(x_test).to(device)
            trottle_output = trottle_model(x_test).to(device)
            test_loss += criterion(angle_output, angle_label)*100
            prediction = angle_output
            #test_loss /= len(test_loader.dataset)
            iter+=1


            if a == 0 and Epoch != 0: # 각 Epoch 마다 한번만 plot하기 위함
                ori_image = sample_batched['image']
                ori_label = sample_batched['label']
                plt.figure(Epoch)
                pltsize = 2
                plt.figure(figsize=(10*pltsize, 10*pltsize))
                for i in range(25):
                     plt.subplot(5, 5, i+1)
                     plt.axis('off')
                     plt.imshow(ori_image[i])
                     plt.rc('font', size=10)
                     plt.title('predict '+ str(round(prediction[i][0].item(),2)) + ' | lable '+ str(round(ori_label[i][0].item(),2)));   

                plt.suptitle('EPOCH : '+str(Epoch))
                plt.savefig(path+str(Epoch) + '.png');            
                a = 1  # Evaluate함수가 다시 수행되기 전까지 plot을 그리지 않음!!
    return test_loss/iter


def infer(model, data, device = 'cpu'):
    # convert_tensor = transforms.ToTensor()
    input = torch.Tensor(data)
    model.eval()
    with torch.no_grad():
        x_input = torch.Tensor(input).float()
        x_input = torch.permute(x_input, (2, 0, 1))
        #print(x_input)
        x_input = x_input.unsqueeze(0)
        x_input = x_input.to(device)
        output = model(x_input).to(device)
    return output


class donkey_dataset(data.Dataset):
    def __init__(self, img_path, record_path):
        self._img = np.load(img_path + "images.npy")
        self._record = np.load(record_path + "labels.npy")

    def __len__(self):
        return self._img.shape[0]

    def __getitem__(self, index):
        img = self._img[index]
        label = self._record[index]
        label = label.tolist()
        label = torch.tensor([float(label[0]), float(label[1])])
        return {'image': img, 'label': label}

'''
class donkey_dataset(data.Dataset):  # donkey dataset generator

    def __init__(self, img_path, record_path):
        self.img_path = img_path
        self.record_path = record_path
    def __len__(self):
        return len(os.listdir(self.img_path))
    def make_file_list(self):
        train_img_list = list()
        image_path = self.img_path
        list_len = len(os.listdir(image_path))

        for img_idx in range(list_len):
            img_path = str(self.img_path) + str(int(str(img_idx)) + 1) + '_cam-image_array_.jpg'
            json_path = str(self.record_path) + 'record_' + str(int(str(img_idx)) + 1) + '.json'
            train_img_list.append([img_path, json_path])
        return train_img_list
    def __getitem__(self, index):
        data_array = self.make_file_list()
        img = cv2.imread(str(data_array[index][0]))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #cv2.imshow("ss",img)
        #cv2.waitKey(0)

        cropped_image = img[104:224, 33:192]/255.
        #print(cropped_image)
        file = open(str(data_array[index][1]), 'r')
        lines = file.readline()
        line=re.findall("-?\d+\.\d+",lines)
        label = torch.Tensor([float(line[0]),float(line[1])]) #angle and throttle
        data = {'image':cropped_image, 'label':label}  
        return data
'''

class BasicBlock(nn.Module): #network model
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(3, 24,
                            kernel_size = 5,
                            stride = 2, 
                            padding = 0, 
                            bias = True)
        #self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 32, 
                            kernel_size = 5, 
                            stride = 2, 
                            padding = 0, 
                            bias = True)
        #self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64,
                            kernel_size = 5,
                            stride = 2, 
                            padding = 0, 
                            bias = True)
        #self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 
                            kernel_size = 3, 
                            stride = 1, 
                            padding = 0, 
                            bias = True)
        #self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 
                            kernel_size = 3, 
                            stride = 1, 
                            padding = 0, 
                            bias = True)
        #self.bn5 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(6656,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,1)

        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)


        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.dropout(out, training = self.training, p = 0.15)
        out = F.relu(self.conv2(out))
        out = F.dropout(out, training = self.training, p = 0.15)
        out = F.relu(self.conv3(out))
        out = F.dropout(out, training = self.training, p = 0.15)
        out = F.relu(self.conv4(out))
        out = F.dropout(out, training = self.training, p = 0.15)
        out = F.relu(self.conv5(out))
        out = F.dropout(out, training = self.training, p = 0.15)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = F.relu(out)
        out = F.dropout(out, training = self.training, p = 0.15)
        out = self.fc2(out)
        out = F.relu(out)
        out = F.dropout(out, training = self.training, p = 0.15)
        
        out = self.fc3(out)
        return out

class BasicBlock2(nn.Module): #network model with bn
    def __init__(self):
        super(BasicBlock2, self).__init__()

        self.conv1 = nn.Conv2d(3, 24,
                            kernel_size = 5,
                            stride = 2, 
                            padding = 0, 
                            bias = True)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 32, 
                            kernel_size = 5, 
                            stride = 2, 
                            padding = 0, 
                            bias = True)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64,
                            kernel_size = 5,
                            stride = 2, 
                            padding = 0, 
                            bias = True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 
                            kernel_size = 3, 
                            stride = 1, 
                            padding = 0, 
                            bias = True)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 
                            kernel_size = 3, 
                            stride = 1, 
                            padding = 0, 
                            bias = True)
        self.bn5 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(6656, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)

        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)


        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.dropout(out, training = self.training, p = 0.15)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.dropout(out, training = self.training, p = 0.15)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.dropout(out, training = self.training, p = 0.15)
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.dropout(out, training = self.training, p = 0.15)
        out = F.relu(self.conv5(out))
        out = F.dropout(out, training = self.training, p = 0.15)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = F.relu(out)
        out = F.dropout(out, training = self.training, p = 0.15)
        out = self.fc2(out)
        out = F.relu(out)
        out = F.dropout(out, training = self.training, p = 0.15)
        
        out = self.fc3(out)
        return out






if __name__ == "__main__":


    EPOCHS = 250
    Batch_size = 256
    learning_rate = 0.0005
    test_size = 1500
    path = './data/'
    save_path = './data/checkpoint/'


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    print('pytorch version : ', torch.__version__, '  device : ', DEVICE)
    data_set = donkey_dataset(img_path = str(path + 'image/'), record_path=str(path+'record/'))
    print('total data_set size is ',len(data_set))
    train_set, test_set = torch.utils.data.random_split(data_set, [len(data_set)-test_size, test_size])
    train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = Batch_size, shuffle = True, num_workers = 1)
    test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = Batch_size, shuffle = True, num_workers = 1, drop_last=True)
    print('train set size is ',len(train_set))
    print('test set size is ',len(test_set))
    for i_batch, sample_batched in enumerate(train_loader):
        print('input image size is', sample_batched['image'].size(), 'and Label size is ', len(sample_batched['label']))
        print('Epochs = ', EPOCHS, ' Batch_size = ', Batch_size, ' Learning_rate : ', learning_rate)
        print('==========================================================')
        print("")
        break################################################################################ for print information

    angle_model=BasicBlock2().to(DEVICE)
    angle_optimizer = torch.optim.AdamW(angle_model.parameters(), lr = learning_rate)
    trottle_model=BasicBlock().to(DEVICE)
    trottle_optimizer = torch.optim.AdamW(trottle_model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss()

    test_loss_prev = 10000
    test_loss_ep = 0

    for Epoch in range(1, EPOCHS + 1): #######################################################for training
        print(Epoch)
        now = datetime.now()
        train(angle_model, trottle_model, angle_optimizer, trottle_optimizer, train_loader, log_interval = 5, device = DEVICE)
        test_loss = evaluate(angle_model, trottle_model, test_loader, DEVICE, Epoch, path=save_path)
        if test_loss < test_loss_prev:
            test_loss_prev = test_loss
            test_loss_ep = Epoch
        print("\n[EPOCH : {}  out of {}], \tTest Loss : {:.8f}, \t \n".format(Epoch, EPOCHS, test_loss))
        torch.save({'angle_model': angle_model.state_dict(), 'trottle_model': trottle_model.state_dict()}, str(save_path + now.strftime("%m-%d-%Y-%H-%M-%S") + str(Epoch)+'_epoch_model.pth'),_use_new_zipfile_serialization=False)
        print("\n Best loss: {:.8f} , in Epoch: {}".format(test_loss_prev, test_loss_ep))

    # ###for inference


    # model.load_state_dict(torch.load(str(path+'41_epoch_model.pth')))
    # path = path + 'test/'
    # file_list = os.listdir(path)
    # file_list_py = [file for file in file_list if file.endswith('.jpg')]
    # print(file_list_py)
    # for i in file_list_py:
    #     test_img = cv2.imread(str('/home/sun/utils/donkey_py/data_set_2/test/'+i))
    #     test_img = test_img[104:224, 35:192]
    #     cv2.imshow('1',test_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     print('===================================')
    #     print(infer(model,test_img,DEVICE))
    #     print('===================================')
    #     input("type enter")
    #     print()
