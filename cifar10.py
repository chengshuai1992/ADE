# -- coding: utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision
import math
import time
import numpy as np
import os
import os.path as osp
import argparse
from margin.AngleLinear import AngleLinear
from margin.ArcMargin import ArcMarginProduct
from margin.MarginCosineProduct import MarginCosineProduct
from torch.autograd import Function
from torchvision import transforms
from torch.autograd import Variable
from cal_map import *
from network import *

from tensorboardX import SummaryWriter
# ====================================Parameter====================================
parser = argparse.ArgumentParser(description='training cifar10')
parser.add_argument('--bits', type=int, default=64, metavar='bts', help='binary bits')
parser.add_argument('--seed', type=int, default=10, help="seed numble")
parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
parser.add_argument('--dataset', type=str, default='cifar10', help="dataset name")
parser.add_argument('--lr', type=float, default='0.001', help="the learning rate")
parser.add_argument('--re', type=float, default='0.3', help="the Regularization coefficient")
parser.add_argument('--classifier_type', type=str, default='AL',
                    help='Which classifier for train. (MCP, AL, ARC, L)')
parser.add_argument('--gpu_id', default=2, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num_class', type=int, default=10,
                    help='number of class')
parser.add_argument('--momentum', type = float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    metavar='W', help='weight decay (default: 0.0005)')

print("count: ", torch.cuda.device_count())
args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
torch.cuda.set_device(args.gpu_id)
topk = [100, 200, 300, 400, 500, 1000]

# fix the random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

writer = SummaryWriter(log_dir='./logs')

# the input config
config = {}
config["snapshot_interval"] = 10  # saving with par 10 epoch
config["hash_bit"] = args.bits
config["dataset"] = args.dataset
config["batch_size"] = args.batch_size
config["num_epochs"] = 50
config["num_classes"] = 10
config["output_path"] = "./model/" + config["dataset"] + "_" + str(config["hash_bit"]) + "bit_/re=" + str(args.re) + '_' + str(args.classifier_type) + '/'
config["snapshot_path"] = config["output_path"] + "Epoch_000" + str(config["num_epochs"]) + "_model.pth.tar"

if not osp.exists(config["output_path"]):
    os.makedirs(config["output_path"])
print("model_path: ", osp.join(config["output_path"],
                                    "last_classifier_model.pth.tar"))

if args.classifier_type != 'L':
    classifier = {
        'MCP': MarginCosineProduct(config["hash_bit"], args.num_class).cuda(),
        'AL' : AngleLinear(config["hash_bit"], args.num_class).cuda(),
        'ARC': ArcMarginProduct(config["hash_bit"], args.num_class).cuda(),
    }[args.classifier_type]
    print("classifier: ", classifier)

# load data
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset
train_dataset = dsets.CIFAR10(root='data/',
                              train=True,
                              transform=train_transform,
                              download=False)

test_dataset = dsets.CIFAR10(root='data/',
                             train=False,
                             transform=test_transform)

database_dataset = dsets.CIFAR10(root='data/',
                                 train=False,
                                 transform=test_transform)

# Construct training, query and database set
X = train_dataset.data
L = np.array(train_dataset.targets)

X = np.concatenate((X, test_dataset.data))
L = np.concatenate((L, np.array(test_dataset.targets)))


# load index
test_index = np.load("./data/cifar-10-batches-py/test.npy")
test_data = X[test_index]
test_L = L[test_index]

data_index = np.load("./data/cifar-10-batches-py/database.npy")
data_set = X[data_index]
dataset_L = L[data_index]

train_index = np.load("./data/cifar-10-batches-py/train.npy")
train_data = X[train_index]
train_L = L[train_index]


train_dataset.data = train_data
train_dataset.targets = train_L
test_dataset.data = test_data
test_dataset.targets = test_L
database_dataset.data = data_set
database_dataset.targets = dataset_L

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=config["batch_size"],
                                           shuffle=True,
                                           num_workers=4)
print("train_loader: ", len(train_loader))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=config["batch_size"],
                                          shuffle=True,
                                          num_workers=4)
print("test_loader: ", len(test_loader))
database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=True,
                                              num_workers=4)
print("database_loader: ", len(database_loader))

# define the network
if args.classifier_type == 'L':
    cnn = hash_SOFTCNN(encode_length=config["hash_bit"], num_classes=config["num_classes"])
else:
    cnn = hash_CNN(encode_length=config["hash_bit"], num_classes=config["num_classes"])

print("cnn: ", cnn)

criterion = torch.nn.CrossEntropyLoss().cuda()

if args.classifier_type == 'L':
    optimizer = torch.optim.SGD([{'params': cnn.parameters()}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.SGD([{'params': cnn.parameters()}, {'params': classifier.parameters()}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

def train_hashCNN(config):
    best = 0.0
    num_epochs = config["num_epochs"]
    start = time.time()

    for epoch in range(num_epochs):
        cnn.cuda()
        cnn.train()

        if args.classifier_type != 'L':
            classifier.train()
        scheduler.step()

        for i, (images, labels) in enumerate(train_loader):
            if i == 0:
                running_loss = 0
                running_loss1 = 0
                running_loss2 = 0

            if i == len(train_loader) - 1:
                title = "train/loss" + "_" + str(args.dataset) + "_" + str(args.classifier_type) + "_reg=" + str(args.re) + "_" + str(args.bits) + "bit"

                writer.add_scalars(title, {"loss": running_loss / i,
                                           "loss1": running_loss1 / i,
                                           "loss2": running_loss2 / i
                                           }, epoch)

            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            # Forward + Backward + Optimize
            if args.classifier_type != 'L':
                feature, code = cnn(images)
                output = classifier(feature, labels)
            else:
                feature, code, output = cnn(images)

            batch, bit = feature.size()
            loss1 = criterion(output, labels)                # CrossEntropyLoss()
            # regulation
            loss2 = torch.mean(torch.abs(torch.pow(torch.abs(feature) - Variable(torch.mean(torch.abs(feature), dim=1, keepdim=True).repeat(1, bit).cuda()), 3)))

            loss = 1.0 * loss1 + args.re * loss2
            running_loss += loss.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if i % (len(train_dataset) // config["batch_size"]) == 0 and i !=0:

                print('Epoch [%d/%d], Iter [%d/%d]  Loss: %.4f  Loss1: %.4f Loss2: %.4f'
                      % (epoch + 1, num_epochs, i, len(train_dataset) // config["batch_size"],
                         running_loss / (i+1), running_loss1 / (i+1), running_loss2 / (i+1)))

        # Test the Model
        cnn.eval()  # Change model to 'eval' mode
        if args.classifier_type != 'L':
            classifier.eval()
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            if args.classifier_type != 'L':
                feature, code = cnn(images)
                output = classifier(feature, labels)
            else:
                feature, code, output = cnn(images)
            _, predicted = torch.max(output.cpu().data, 1)

            total += labels.size(0)
            correct += (predicted == labels.cpu().data).sum()


        print('Test Accuracy of the model: %.2f %%' % (100.0 * correct / total))

        if 100.0 * correct / total > best:
            best = 100.0 * correct / total
        print('best: %.2f %%' % (best))

    time_elapsed = time.time() - start
    print('Train complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def binary_output(dataloader):
    use_cuda = torch.cuda.is_available()

    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()
    # use the .eval() will remove the dropout and batch
    cnn.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)

        if args.classifier_type != 'L':
            feature, code = cnn(inputs)
        else:
            feature, code, _ = cnn(inputs)
        full_batch_output = torch.cat((full_batch_output, code.data), 0)
        full_batch_label = torch.cat((full_batch_label, targets.data), 0)
    return torch.sign(full_batch_output), full_batch_label

def test():
    cnn.cuda()
    cnn.eval()

    retrievalB, retrievalL = binary_output(database_loader)
    print("binary_output1 finished")
    queryB, queryL = binary_output(test_loader)
    print("binary_output2 finished")

    print("retrievalB: ", np.shape(retrievalB))
    print("retrievalL: ", np.shape(retrievalL))
    print("queryB: ", np.shape(queryB))
    print("queryL: ", np.shape(queryL))

    print('---calculate map---')
    map = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
    print("map: ", map)
    r_map = calculate_map_radius(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, r=2)
    print("r_map: ", r_map)
    pre_radius_map = calculate_pre_radius(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, r=2)
    print("pre_radius_map: ", pre_radius_map)
    for topk_ in topk:
        topk_pre = calculate_pre_topk(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=topk_)
        print("topk_pre: ", topk_pre)
        print("topk_: ", topk_)
    print("bit: ", config["hash_bit"])


if __name__ == "__main__":
    print('Start training the model')
    train_hashCNN(config)
    if args.classifier_type == 'L':
        torch.save(cnn.state_dict(), osp.join(config["output_path"],
                                              "last_cnn_model.pth.tar"))
    else:
        torch.save(cnn.state_dict(), osp.join(config["output_path"],
                                              "last_cnn_model.pth.tar"))
        torch.save(classifier.state_dict(), osp.join(config["output_path"],
                                              "last_classifier_model.pth.tar"))

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    print('Start testing the model')
    test()
