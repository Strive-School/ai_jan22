from sched import scheduler
from numpy import fromfunction
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim


from sample_convolution import Convolution
from conv_data_handler import train_data, test_data

trainloader = T.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = T.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)


model = Convolution()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
sched = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

epochs = 10

train_losses = []
test_losses = []

for epoch in range(epochs):

    running_train_loss = 0
    running_test_loss = 0

    for train_imgs, train_labels in iter(trainloader):
        print('training...')

        optimizer.zero_grad()

        pred = model.forward(train_imgs)
        pred = pred.squeeze()
        # print(pred.shape, train_labels.shape)
        train_labels = train_labels.float()

        train_loss = criterion(pred, train_labels)

        train_loss.backward()

        optimizer.step()

        running_train_loss += train_loss.item()

    model.eval()
    with T.no_grad():

        for test_imgs, test_labels in iter(testloader):
            print('evaluating...')

            test_pred = model.forward(test_imgs)
            test_pred = test_pred.squeeze()

            test_labels = test_labels.float()


            test_loss = criterion(test_pred, test_labels)

            running_test_loss += test_loss.item()
    model.train()
    
    avg_train_loss  = running_train_loss/len(trainloader)
    avg_test_loss   = running_test_loss/len(testloader)

    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)

    print(f'Epoch: {epoch+1} | train loss: {avg_train_loss} | test loss: {avg_test_loss}')

    sched.step()

plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')

plt.show()


        



