from turtle import forward
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Recurrent_Net(nn.Module):

    def __init__(self, in_size, hid_size, n_layers, batch_size, seq_length):
        super(Recurrent_Net, self).__init__()
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.hidden_size = hid_size

        self.rnn = nn.RNN(in_size, hid_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hid_size, seq_length)

    def forward(self, x):

        h_0 = T.zeros((self.n_layers, self.batch_size, self.hidden_size))

        _, h_n = self.rnn(x, h_0)
        last_hidden = h_n[-1]
        out = F.relu(self.fc(last_hidden))

        return out


batch_size = 32
seq_length = 20
hid_size = 15
in_size = 8
n_layers = 100


fake_seq = T.rand((batch_size, seq_length, in_size))

model = Recurrent_Net(in_size, hid_size, n_layers, batch_size, seq_length)

out = model.forward(fake_seq)

print(out.shape)