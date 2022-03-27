import numpy as np
import pandas
import matplotlib.pyplot as plt
from PIL import Image

class Net:

    def __init__(self, m, L, mode):
        self.X = []
        self.S = []
        self.L = L
        self.sz = [2, m, 1]
        self.W = [None]
        self.mode = mode
        self.grad = [None for i in range(self.L + 1)]

        for i in range(self.L):
            l, r = self.sz[i], self.sz[i + 1]
            w = np.random.randn(l + 1, r)
            self.W.append(w)

    def reset(self):
        self.W[0] = None
        for i in range(self.L):
            l, r = self.sz[i], self.sz[i + 1]
            w = np.ones([l + 1, r]) * 0.25
            self.W[i + 1] = w

    def forward(self, X_data):
        self.X = []
        self.S = []
        one = np.ones([1, 1])
        X_data = np.vstack((one, X_data))
        self.X.append(X_data)
        self.S.append(None)
        for l in range(1, self.L):
            s = np.matmul(self.W[l].T, self.X[l - 1])
            self.S.append(s)
            one = np.ones([1, 1])
            x = np.vstack((one, np.tanh(s)))
            self.X.append(x)

        s = np.matmul(self.W[self.L].T, self.X[self.L - 1])
        self.S.append(s)
        one = np.ones([1, 1])
        if self.mode == 'identity':
            x = np.vstack((one, s))
        elif self.mode == 'tanh':
            x = np.vstack((one, np.tanh(s)))
        elif self.mode == 'sign':
            x = np.vstack((one, np.sign(s)))

        self.X.append(x)
        out = self.X[-1][1:]
        return out
            
    def backward(self, X_data, y):
        out = self.forward(X_data)
        grad_out = None
        if self.mode == 'identity':
            grad_out = 1
        elif self.mode == 'tanh':
            grad_out = 1 - np.power(self.X[self.L][1:], 2)
        elif self.mode == 'sign':
            grad_out = 0

        self.grad[self.L] = 2 * (self.X[self.L][1:] - y) * grad_out

        for l in range(self.L - 1, 0, -1):
            theta = (1 - self.X[l] * self.X[l])[1:]
            self.grad[l] = theta * (np.matmul(self.W[l + 1], self.grad[l + 1])[1:])
        
        return self.grad

    def cal_error(self, X, y1, X_val = None, y_val=None):
        N = X.shape[0]
        d = X.shape[1]
        Ein = 0
        Eval = 0

        self.grad_W = [None]

        for i in range(1, self.L + 1):
            self.grad_W.append(np.zeros(self.W[i].shape))

        for i in range(N):
            x = X[i, :]
            x = x.reshape(1, -1)
            Ein += np.power(self.forward(x.T) - y1[i], 2)/(4 * N)
            grad = self.backward(x.T, y1[i])
            for l in range(1, self.L + 1):
                self.grad_W[l] += np.matmul(self.X[l - 1], self.grad[l].T)/(4 * N)
        #for l in range(1, self.L + 1):
        #    Ein += 0.01/N * np.sum(self.W[l] * self.W[l])

        M = X_val.shape[0]
        for i in range(M):
            x = X_val[i, :]
            x = x.reshape(1, -1)
            Eval += np.power(self.forward(x.T) - y_val[i], 2)/(4 * M)
    
        return Ein, Eval, self.grad_W

def yfeature(y):
    y_feature = np.zeros([y.shape[0], 1])
    for i in range(y.shape[0]):
        if y[i] == 1:
            y_feature[i] = 1
        else:
            y_feature[i] = -1
    return y_feature

def feature1(x):
    ave_intensity = np.sum(x)/(16 * 16)
    return ave_intensity

def feature2(x):
    sum = 0
    x = x.reshape([16, 16])
    for i in range(16):
        for j in range(8):
            if np.abs(x[i, j] - x[i, 15 - j]) > 0.1:
                sum = sum + 1
    return sum/(16 * 16)

def main():
    train_address = "ZipDigits.train"
    test_address = "ZipDigits.test"
    
    # Input
    X_train = pandas.read_csv(open(train_address), delimiter=" ")
    X_train = X_train.to_numpy()
    y_train = X_train[:, 0]
    y_train = y_train.reshape(X_train.shape[0], 1)
    X_train = X_train[:, 1:-1]
    y_train_feature = yfeature(y_train)
    
    X_test = pandas.read_csv(open(test_address), delimiter=" ")
    X_test = X_test.to_numpy()
    y_test = X_test[:, 0]
    y_test = y_test.reshape(X_test.shape[0], 1)
    X_test = X_test[:, 1:-1]
    y_test_feature = yfeature(y_test)

    X = np.vstack((X_train, X_test))
    y_feature = np.vstack((y_train_feature, y_test_feature))
    X_feature = np.zeros([X.shape[0], 2])

    for i in range(X.shape[0]):
        X_feature[i, 0] = feature1(X[i])
        X_feature[i, 1] = feature2(X[i])

    min1 = np.min(X_feature[:, 0])
    max1 = np.max(X_feature[:, 0])
    X_feature[:, 0] = (2 * (X_feature[:, 0] - min1)/(max1 - min1)) - 1

    min2 = np.min(X_feature[:, 1])
    max2 = np.max(X_feature[:, 1])
    X_feature[:, 1] = (2 * (X_feature[:, 1] - min2)/(max2 - min2)) - 1

    X_feature_perm = np.zeros((X_feature.shape[0], 2))
    y_feature_perm = np.zeros((y_feature.shape[0], 1))

    perm = np.random.permutation(X_feature.shape[0])
    for i in range(X_feature.shape[0]):
        X_feature_perm[i, :] = X_feature[perm[i], :]
        y_feature_perm[i, :] = y_feature[perm[i], :]

    training = X_feature_perm[0:250]
    training_y = y_feature_perm[0:250]
    validation = X_feature_perm[250:300]
    validation_y = y_feature_perm[250:300]
    testing = X_feature[300:1000]
    testing_y = y_feature_perm[300:1000]

    net = Net(10, 2, "identity")
    #for l in range(1, net.L + 1):
    #    for i in range(net.W[l].shape[0]):
    #        for j in range(net.W[l].shape[1]):
    #            net.reset()
    #            net.W[l][i, j] += 0.0001
    #            Ein, grad_W = train(net, X, y1)
    #            print(grad_W)
    #            net.W[l][i, j] -= 0.0001

    iters = []
    Eins = []
    Evals = []
    Etests = []
    eta = 0.0001
    alpha = 1.05
    beta = 0.8
    batch_size = 5
    iterations = 10000
    start = 0

    for i in range(iterations):
        end = start + batch_size
        Ein, Eval, grad_W = net.cal_error(training[start: end, :], training_y[start: end], testing[start: end, :], testing_y[start: end])
        for l in range(1, net.L + 1):
            net.W[l] -= eta * grad_W[l]
        Ein_new, Eval_new, grad_W_new = net.cal_error(training[start: end, :], training_y[start: end], testing[start: end, :], testing_y[start: end])
        if Ein_new < Ein:
            eta = alpha * eta
        else:
            for l in range(1, net.L + 1):
                net.W[l] += eta * grad_W[l]
            eta = beta * eta
        if i % 100 == 0:
            print("Iter: ", i)
            iters.append(i)
            Eins.append(np.log(Ein[0][0]))
            Evals.append(Eval[0][0])
        if end >= training.shape[0]:
            start = 0
        else:
            start = end

    net.mode = "sign"
    ans = 0
    for i in range(testing.shape[0]):
        out = net.forward(testing[0].reshape(-1, 1))
        ground_truth = testing_y[i]
        if out == ground_truth:
            ans += 1
    print("Acc: ",ans/testing.shape[0])
    print("Etest: ", Evals[-1])

    #plt.plot(iters, Eins, 'b', label='Eins-iters')
    #plt.plot(iters, Evals, 'r', label='Evals-iters')
    #plt.show()

    
    min11 = np.min(training[:, 0])
    max11 = np.max(training[:, 0])
    min22 = np.min(training[:, 1])
    max22 = np.max(training[:, 1])
    grid = np.zeros((500, 500))
    
    #for i in range(0, 500):
    #    for j in range(0, 500):
    #        x1 = i * (max11 - min11)/500 + min11
    #        x2 = j * (max22 - min22)/500 + min22
    #        X = np.asarray([x1, x2]).reshape(-1, 1)
    #        grid[i, j] = net.forward(X)[0][0]
    #
    #img_np = Image.fromarray(grid.astype('uint8')).convert('RGB')
    #img_np.show()

if __name__ == '__main__':
    main()




