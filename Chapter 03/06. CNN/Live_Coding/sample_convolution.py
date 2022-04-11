import torch as T
import torch.nn as nn
import torch.nn.functional as F

# final_dim = (start_dim - k + 2*p) // s + 1


class Convolution(nn.Module):

    def __init__(self):
        super(Convolution, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5, 2) # (244 - 5 + 2 * 0) // 2 + 1 = 120
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(128, 64, 5, 2, 2) 
        self.conv3 = nn.Conv2d(64, 32, 3, 2) 
        self.fc1 = nn.Linear(32 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)
    
    def forward(self, x):
        # x = x.float()
        x = F.relu(self.conv1(x)) # (244 - 5 + 2 * 0) // 2 + 1 = 120
        x = self.pool(x) # (120 - 3) // 2 + 1
        x = F.relu(self.conv2(x)) # (120 - 5 + 2 * 2) // 2 + 1 = 60
        x = self.pool(x)
        x = F.relu(self.conv3(x)) # (60 - 3 + 2 * 0) // 2 + 1 = 29
        # print(x.shape)
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return T.sigmoid(self.out(x))

# model = Convolution()

# fake_imgs = T.rand((32, 3, 100, 100))

# criterion = nn.CrossEntropyLoss()

# logits = model.forward(fake_imgs)

# ps = F.softmax(logits, dim=1)

# print(ps.argmax(dim=1))
# print(ps.argmax(dim=1).shape)




