from cgi import test
import torch as T
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt

df = pd.read_csv('data.csv', header=None)

print(df.head())

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, 32)  # input = num of features, out up to you
        self.hidden1 = nn.Linear(32, 16) # input = out of previus layer, output up to you
        self.hidden2 = nn.Linear(16, 8) # input = out of previus layer, output up to you
        self.output = nn.Linear(8, 1) # input = out of previus layer, it depends on the task
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        first_layer = self.input_layer(x)
        act1 = self.sigmoid(first_layer)
        second_layer = self.hidden1(act1)
        act2 = self.sigmoid(second_layer)
        third_layer = self.hidden2(act2)
        act3 = self.sigmoid(third_layer)
        out_layer = self.output(act3)
        # prediction = self.sigmoid(out_layer)
        return self.sigmoid(out_layer)

model = Classifier()

criterion = nn.BCELoss()
optimizer = T.optim.Adam(model.parameters(), lr=1e-3) # 0.001 1 * 10^-3

X = T.from_numpy(df[[0, 1]].values).float()
y = T.from_numpy(df[[2]].values).float()

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=73)

epochs = 1000

train_losses = []
test_losses = []
accuracies = []

for epoch in range(epochs):

    optimizer.zero_grad()   # 1st step: reset the gradients

    pred = model.forward(x_train) #2nd step: make the prediction

    train_loss = criterion(pred, y_train)   #3rd step: compute the loss

    train_loss.backward() #4th step: backward pass

    optimizer.step() #5th step: save the weights

    model.eval()
    with T.no_grad():
        test_pred = model.forward(x_test)

        test_loss = criterion(test_pred, y_test)

        classes = test_pred > 0.5

        acc = sum(classes == y_test) / classes.shape[0]

    model.train()

    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    accuracies.append(acc)
    print(f'Epoch: {epoch + 1} | loss: {train_loss.item()} | test loss: {test_loss.item()} | accuracy: {acc}')


plt.plot(train_losses, label='train Loss')
plt.plot(test_losses, label='test Loss')
plt.plot(accuracies, label='accuracy')
plt.legend()
plt.show()












    

