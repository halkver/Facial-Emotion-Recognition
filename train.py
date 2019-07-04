from __future__ import print_function
import utils

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from dataloader import Dataset
from model import vgg19

total_epochs = 300
cut_size = 44
batch_size = 16

use_cuda = torch.cuda.is_available()

transforms_train = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transforms_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = Dataset(set_type="Training", transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

publicset = Dataset(set_type="PublicTest", transform=transforms_test)
publicloader = torch.utils.data.DataLoader(publicset, batch_size=batch_size, shuffle=False, num_workers=1)

privateset = Dataset(set_type="PrivateTest", transform=transforms_test)
privateloader = torch.utils.data.DataLoader(privateset, batch_size=batch_size, shuffle=False, num_workers=1)

model = vgg19()
if use_cuda: torch.nn.DataParallel(model).cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
milestones = [total_epochs*.5, total_epochs*.75, total_epochs*.9]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

def train(epoch):
    model.train()
    print('\nEpoch: {}'.format(epoch))
    running_loss = 0.0
    scheduler.step()
    correct, total = 0, 0
    for param_group in optimizer.param_groups:
        print('Learning rate:',param_group['lr'])
    for i, (inputs, targets) in enumerate(trainloader):
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        summary_string = 'Loss :{:.3f} | Acc :{:.3f}% |({:d}/{:d})'.format(running_loss/(i+1), 100.*correct/total, correct, total)
        utils.progress_bar(i, len(trainloader), summary_string)
    print('Scheduler loss check: {:f}'.format(running_loss/(i+1)))
    return 100.*correct/total

def privateTest(best_test):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(privateloader):
            bs, ncrops, c, h, w = inputs.size()
            if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
            inputs = inputs.view(-1, c, h, w)
            outputs = model(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            summary_string = 'Acc :{:.3f}% |({:d}/{:d})'.format(100.*correct/total, correct, total)
            utils.progress_bar(i, len(trainloader), summary_string)

    acc = 100.*correct/total
    if acc>best_test:
        state = {
            'model': model.state_dict() if use_cuda else model,
            'acc': acc
        }
        torch.save(state, 'model.pt')
        print('New best: {:.3f}% Saving...'.format(acc))
        return acc
    else:
        return False

if __name__=='__main__':

    best_test = 0
    for epoch in range(total_epochs):
         train_acc = train(epoch)
         res = privateTest(best_test)
         if res: best_test = res

    print('Training accuracy: {:.3f}'.format(train_acc))
    print('Test accuracy: {:.3f}'.format(best_test))
