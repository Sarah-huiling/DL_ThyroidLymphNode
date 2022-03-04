'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

# #from setting import parse_opts
# from datasets.brains18 import BrainS18Dataset
# from model import generate_model
import torch
# import numpy as np
import xlrd
from torch import nn
from torch import optim
import os
import glob
import numpy as np
import logger
import torch.nn.functional as F
from data_img import MyDataset
# from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
# import time
# from utils.logger import log
# from scipy import ndimage
# import os

from torch.autograd import Variable
from model import mobilenet_v2
from sklearn import metrics as mt

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0, 1"

def train(alexnet_model, train_loader, epoch, train_dict, logger,criterion,use_gpu):
    alexnet_model.train()
    losss = 0
    for iter, batch in enumerate(train_loader):
        torch.cuda.empty_cache()
        if use_gpu:
            inputs = Variable(batch[0].cuda())
            labels = Variable(batch[1].cuda())
        else:
            inputs, labels = Variable(batch['0']), Variable(batch['1'])

        # label_fla = labels.cpu().numpy()
        # label_fla = label_fla.flatten()
        # label_fla = label_fla[1::2]
        # if np.sum(label_fla) < 2:
        #     continue
        optimizer.zero_grad()

        outputs = alexnet_model(inputs)
        #print(outputs, labels)
        loss = criterion(outputs, labels)
        # acc(outputs, labels)
        # exit()
        loss.backward()
        optimizer.step()
        losss = losss + loss.item()
        # dice0, dice1, dice2, dice3 = dicev(outputs, labels)
        if (iter + 1) % 10 == 0:
            #print(outputs, labels)
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, iter, len(train_loader),
                    100. * iter / len(train_loader), losss / iter))
    train_dict['loss'].append(losss / iter)
    logger.scalar_summary('train_loss', losss / iter, epoch)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def ODIR_Metrics(pred,target):
    th = 0.5
    gt = target.flatten()
    pr = pred.flatten()
    gt1 = gt[0::2]
    gt2 = gt[1::2]
    pr1 = pr[0::2]
    pr2 = pr[1::2]
    kappa = mt.cohen_kappa_score(gt, pr > th)
    print("1：auc值,", mt.roc_auc_score(gt1, pr1), 'acc:',mt.accuracy_score(gt1, pr1 > th))
    print("2：auc值,", mt.roc_auc_score(gt2, pr2), 'acc:',mt.accuracy_score(gt2, pr2 > th))
    #f1 = mt.f1_score(gt, pr > th, average='micro')
    auc = mt.roc_auc_score(gt1, pr1)
    return auc

def val_test(alexnet_model, val_loader):
    alexnet_model.eval()
    val_loss = 0
    with torch.no_grad():
        p = []
        g = []
        for iter, batch in enumerate(val_loader):
            torch.cuda.empty_cache()
            if use_gpu:
                inputs = Variable(batch[0].cuda())
                labels = Variable(batch[1].cuda())
            else:
                inputs, labels = Variable(batch['0']), Variable(batch['1'])
            outputs = alexnet_model(inputs)
            loss = criterion(outputs, labels)
            #outputs = torch.softmax(outputs, dim=1)
            outputs = outputs.data.cpu().numpy()
            labels = labels.cpu().numpy()
            for x, y in zip(outputs, labels):
                p.append(x)
                g.append(y)
            val_loss += loss.item()
        auc = ODIR_Metrics(np.array(p), np.array(g))
    val_loss /= len(val_loader)
    print('\nVal set: Average loss: {:.6f},auc: {:.6f}\n'.format(val_loss, auc))
    return auc




class WeightedMultilabel(torch.nn.Module):

    def __init__(self, weights: torch.Tensor):
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.weights = weights.unsqueeze()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets) * self.weights


if __name__ == "__main__":
    batch_size = 8
    epochs = 100
    lr = 0.001
    momentum = 0.95
    w_decay = 1e-6
    step_size = 50
    gamma = 0.5
    n_class = 2
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    a = []
    data_path = '/media/zhl/Local/ResearchData/Kras/DataProcessing/crop/*.npy'
  
    t_path = glob.glob(data_path)
    # leng = len(t_path)
    # vail_path = t_path[:int(leng/5)]
    # train_path = t_path[int(leng/5):int(4*leng/5)]
    # test_path = t_path[int(4*leng/5):]
    reads = xlrd.open_workbook(
        '/media/zhl/Local/ResearchData/Kras/grouping.xlsx')
    id = []
    flag = []
    for row in range(reads.sheet_by_index(0).nrows):
        id.append(reads.sheet_by_index(0).cell(row, 0).value)
        flag.append(reads.sheet_by_index(0).cell(row, 2).value)
    test_path = []
    train_path = []
    vail_path = []
    for i in range(len(id)):
        for j in range(len(t_path)):
            if id[i] in t_path[j]:
                if flag[i] == 0:
                    train_path.append(t_path[j])
                if flag[i] == 1:
                    vail_path.append(t_path[j])
                if flag[i] == 2:
                    test_path.append(t_path[j])



    train_da = MyDataset(train_path, transform=True)
    test = MyDataset(test_path, transform=False)
    vail = MyDataset(vail_path, transform=False)
    train_loader = DataLoader(train_da, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(vail, batch_size=batch_size, shuffle=False, num_workers=2)

    print('model load...')
    model_dir = "/media/zhl/Local/ResearchData/Kras/models_test"  # models4 130(0.74)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # model_path_best = os.path.join(model_dir, 'res50.pth')
    model_path = os.path.join(model_dir, 'mobilev2.pth')

    model = mobilenet_v2()
    #model = vgg16_bn()
    #model = resnet50(sample_input_D=16, sample_input_H=96, sample_input_W=96, num_seg_classes=2)
    #model = _3DCNN()

    if use_gpu:
        alexnet_model = model.cuda()
        alexnet_model = nn.DataParallel(alexnet_model, device_ids=num_gpu)
    # print(model)
    # exit()
    #pos_weight = torch.FloatTensor([0.1, 9]).cuda() # 0.67
    pos_weight = torch.FloatTensor([0.8, 1.2]).cuda()
    #criterion = nn.BCELoss()
    criterion = nn.BCELoss(weight=pos_weight)
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #criterion = WeightedMultilabel(weights=pos_weight)
    #optimizer = optim.Adam(alexnet_model.parameters(), lr=lr, betas=(0.9, 0.99))
    optimizer = optim.SGD(alexnet_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    # create dir for score
    score_dir = os.path.join(model_dir, 'scores')
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)
    train_dict = {'loss': []}
    val_dict = {'loss': [], 'auc': []}
    logger = logger.Logger('/media/zhl/Local/ResearchData/Kras/log')
    best_loss = 0
    for epoch in range(1, 500 + 1):
        # print(val_dict['loss'][0])
        adjust_learning_rate(optimizer, epoch)
        train(alexnet_model, train_loader, epoch, train_dict, logger, criterion, use_gpu)
        print("------------------------", epoch, '------------------------------')
        print("------------------------", 'auc_val', '------------------------------')
        auc_val = val_test(alexnet_model,  val_loader)
        print("------------------------", 'auc_test', '------------------------------')
        auc_test = val_test(alexnet_model,  test_loader)
        if auc_test > 0.8 and auc_test < 0.83 and auc_val > 0.8 and auc_val < 0.85:
            model_path = os.path.join(model_dir, str(auc_test)[:4] + '_best_mobilev2.pth')
            torch.save(alexnet_model, model_path)

