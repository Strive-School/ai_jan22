from os import pardir
import pandas as pd
import numpy as np

data = pd.read_csv("climate.csv")
data = data.drop(["Date Time"], axis=1)

def pairing(data, seq_len=6):

    x = []
    y = []

    for i in range(0,(data.shape[0] - seq_len+1), seq_len+1 ): # range is reduced by the len of seq + 1 so that we do not go out of bounds
                                                            # we step for that same amount of steps as the seq_len
        seq = np.zeros( (seq_len, data.shape[1]) ) #creating a matrix of zeros with the shape of seq_len and the number of columns of the data
        
        for j in range(seq_len):  # filling the matrix with the data

            seq[j] = data.values[i+j]  # filling the matrix with the data we add i to make the jump of the seq_len

        x.append(seq.flatten())  # flattening the matrix and appending it to the x list
        y.append( data["T (degC)"][i+seq_len] )  # appending the target to the y list 

    return np.array(x), np.array(y)

print(data.shape)

x, y = pairing(data)

print(x.shape)

print(y[0])
print(y[1])

print("heheh")