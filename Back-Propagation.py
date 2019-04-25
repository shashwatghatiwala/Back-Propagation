from random import random
import numpy as np
import math
import matplotlib.pyplot as plt

trainingdata = 'hw3trainingdata.csv'
raw_data = open(trainingdata, 'rt', encoding="utf-8")
data = np.genfromtxt(raw_data, delimiter = ' ', dtype = 'float')

def initialize_weights(h):
    a = 2
    weights_input_hidden = [[0 for i in range(a)] for j in range(h)]
    for i in range(h):
        for j in range(a):
            weights_input_hidden[i][j] = random()
    return weights_input_hidden

def activation(f):
    x = (1 / (1 + math.exp(-f)))
    return x

def forward_pass_hidden(n, k):
    p = len(n)
    q = len(n[0])
    weighted_sum_input = [[0 for j in range(p)] for i in range(k)]
    sum_input = 0
    for l in range(k):
        for i in range(p):
            for j in range(q):
                if(j == 0):
                    sum_input = 1*n[i][j]
                else:
                    sum_input = sum_input + data[l][0]*n[i][j]
            weighted_sum_input[l][i] = activation(sum_input)
            sum_input = 0
    return weighted_sum_input

hidden_layer = 20
#     input("Number of Hidden Neurons: ")
# hidden_layer = int(hidden_layer)
r = hidden_layer + 1
data_length = len(data)

weights_hidden_output = [0 for i in range(r)]
for i in range(r):
    weights_hidden_output[i] = random()
print("Initial Weights_IJ: ", weights_hidden_output)

def forward_pass_output(n):
    p = len(n)
    r = hidden_layer + 1

    weighted_sum_output = [0 for i in range(p)]
    sum_output = 0
    for l in range(p):
        for j in range(r):
            if(j == 0):
                sum_output = weights_hidden_output[j]
            else:
                sum_output = sum_output + n[l][j-1]*weights_hidden_output[j]

        weighted_sum_output[l] = activation(sum_output)
        sum_output = 0
    return weighted_sum_output


W_I = initialize_weights(hidden_layer)
print("Initial Weights_JK: ", W_I)

J = [0 for i in range(data_length)]

Total_Error = 4
Delta_I = [0 for i in range(data_length)]
Delta_J = [[0 for i in range(hidden_layer)] for j in range(data_length)]
Y = [0 for i in range(data_length)]

H = [[0 for i in range(hidden_layer)]for j in range(data_length)]
H = forward_pass_hidden(W_I, data_length)
print("H_J: ", H)
Y = forward_pass_output(H)
print("Y_Hat: ", Y)

for k in range(data_length):
    Delta_I[k] = (data[k][1] - Y[k]) * (Y[k]) * (1 - Y[k])
print("Delta_I: ", Delta_I)

for k in range(data_length):
    for i in range(hidden_layer):
        Delta_J[k][i] = (Delta_I[k] * weights_hidden_output[i + 1]) * (H[k][i]) * (1 - H[k][i])
        Delta_J[k][i] = (Delta_I[k] * weights_hidden_output[i + 1]) * (H[k][i]) * (1 - H[k][i])
print("Delta_J: ", Delta_J)

Loss = []
E = []
epoch = 0
threshold = 0.05
while(Total_Error > threshold):
    Total_Error = 0
    for k in range(data_length):
        #Updating Weights
        step = 0.1
        for i in range(hidden_layer + 1):
            if(i == 0):
                weights_hidden_output[i] = weights_hidden_output[i] + (step)*(Delta_I[k])
            else:
                weights_hidden_output[i] = weights_hidden_output[i] + (step)*(Delta_I[k])*((H[k][i-1]))

        for i in range(hidden_layer):
            for j in range(2):
                if(j == 0):
                    W_I[i][j] = W_I[i][j] + step*Delta_J[k][j]*1
                elif(j != 0):
                    W_I[i][j] = W_I[i][j] + step*Delta_J[k][j]*data[k][0]
        #Computing Total Error
        J[k] = (Y[k] - data[k][1]) ** 2
        Total_Error += J[k]
        #Forward Pass for Input to Hidden
        H = forward_pass_hidden(W_I, data_length)

        # Forward Pass for Hidden to Output
        Y = forward_pass_output(H)

        #Delta_I Computation
        Delta_I[k] = (data[k][1] - Y[k]) * (Y[k]) * (1 - Y[k])

        #Delta_J Computation
        for i in range(hidden_layer):
            Delta_J[k][i] = (Delta_I[k] * weights_hidden_output[i + 1]) * (H[k][i]) * (1 - H[k][i])
            Delta_J[k][i] = (Delta_I[k] * weights_hidden_output[i + 1]) * (H[k][i]) * (1 - H[k][i])
        epoch += 1
    E.append(epoch)
    Loss.append(Total_Error)
    print("Total Error: ", Total_Error)
print("Final Total Error: ", Total_Error)
print("Number of Epochs: ", epoch)
print("W_JK: ", W_I)
print("W_IJ: ", weights_hidden_output)

plt.plot(E, Loss)
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.show()

def prediction(h, w1, w2):
    testingdata = "hw3testingdata.csv"
    raw_data = open(trainingdata, 'rt', encoding="utf-8")
    testing_data = np.genfromtxt(raw_data, delimiter=' ', dtype='float')

    test_data_length = len(testing_data)
    test_J = [0 for i in range(test_data_length)]
    test_Forward_input_hidden = [[0 for i in range(h)] for j in range(test_data_length)]
    test_Forward_hidden_output = [0 for i in range(test_data_length)]
    Testing_Error = 0
    for k in range(test_data_length):
        test_Forward_input_hidden = forward_pass_hidden(w1, test_data_length)
        test_Forward_hidden_output = forward_pass_output(test_Forward_input_hidden)
        test_J[k] = (Y[k] - testing_data[k][1]) ** 2
        Testing_Error += test_J[k]
    print("Testing_Error: ", Testing_Error)

    # y_x = [0 for i in range(test_data_length)]
    # for k in range(test_data_length):
    #     y_x[k] = testing_data[k][1]
    # plt.plot(test_Forward_hidden_output)
    # plt.plot(y_x)
    # plt.show()

prediction(hidden_layer, W_I, weights_hidden_output)











