import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import ZS

train_data=torchvision.datasets.CIFAR10("data",train=True,transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data=torchvision.datasets.CIFAR10(root="data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
train_data_size=len(train_data)
test_data_size=len(test_data)
print("size of train data:{}".format(train_data_size))
print("size of test data:{}".format(test_data_size))

train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

zs=ZS()
loss_fn=nn.CrossEntropyLoss()
learning_rate=1e-2
optimizer=torch.optim.SGD(zs.parameters(),lr=learning_rate)
total_train_step=0
total_test_step=0
epoch=10
writer=SummaryWriter("logs_train")
for i in range(epoch):
    print("第{}轮训练开始——".format(i+1))
    zs.train()
    for data in train_dataloader:
        imgs,targets=data
        output=zs(imgs)
        loss=loss_fn(output,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step=total_train_step+1
        if total_train_step%100==0:
            print("训练次数：{},Loss:{}".format(total_train_step,loss))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    zs.eval()
    total_test_lost=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            outputs=zs(imgs)
            loss=loss_fn(outputs,targets)
            total_test_lost+=loss.item()
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy+=accuracy
    print("整体测试集上的loss：{}".format(total_test_lost))
    print("Accuracy:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_lost,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step+=1

    torch.save(zs,"zs_{}.pth".format(i))
    print("模型已保存")
writer.close()

