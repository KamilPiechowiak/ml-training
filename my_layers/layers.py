#my own layers and optimizers implementation
import numpy as np

class Updater:
    def computeGradient(self, dW):
        return dW

class Momentum(Updater):
    def __init__(self, beta, shape):
        self.beta = beta
        self.v = np.zeros(shape)
    def computeGradient(self, dW):
        self.v = self.beta*self.v + dW
        return self.v

class Adam(Updater):
    def __init__(self, beta1, beta2, shape):
        self.beta1, self.beta2 = beta1, beta2
        self.v = np.zeros(shape)
        self.cache = np.zeros(shape)
        self.t = 1
        self.eps = 1e-8
    def computeGradient(self, dW):
        self.v = self.beta1*self.v + (1-self.beta1)*dW
        self.cache = self.beta2*self.cache + (1-self.beta2)*dW**2
        vt = self.v/(1-self.beta1**self.t)
        cachet = self.cache/(1-self.beta2**self.t)
        self.t+=1
        return vt/(np.sqrt(cachet)+self.eps)

class Layer:
    def forward(self, x):
        pass
    def backward(self, x):
        pass
    def learn(self, learning_rate, weight_decay):
        pass
    

class ReLU(Layer):
    def forward(self, x):
        self.x = x
        self.y = np.maximum(0, x)
        return self.y
    def backward(self, dy):
        self.dy = dy
        self.dx = self.dy
        self.dx[self.x <= 0] = 0
        return self.dx

class FC(Layer):
    def __init__(self, A, B):
        self.W = np.random.randn(A, B)*np.sqrt(2/A)
        # self.updater = Momentum(0.9, self.W.shape)
        # self.updater = Updater()
        self.updater = Adam(0.9, 0.999, self.W.shape)
    def forward(self, x):
        self.x = x
        self.y = np.dot(self.x, self.W)
        return self.y
    def backward(self, dy):
        self.dy = dy
        self.dW = np.dot(self.x.T, self.dy)/self.dy.shape[0]
        self.dx = np.dot(self.dy, self.W.T)
        return self.dx
    def learn(self, learning_rate, weight_decay):
        self.W+= -learning_rate*self.updater.computeGradient(self.dW) - weight_decay*self.W

class Softmax(Layer):
    def forward(self, x):
        self.x = x
        x_max = np.max(x)
        self.y = np.exp(self.x-x_max)/np.sum(np.exp(self.x-x_max), axis=1, keepdims=True)
        return self.y
    def backward(self, dy):
        self.dx = self.y
        self.dx[range(self.dx.shape[0]),dy]-=1
        return self.dx

class Flatten(Layer):
    def forward(self, x):
        self.shape = x.shape
        n = self.shape[0]
        return x.reshape(n,-1)
    def backward(self, dy):
        return dy.reshape(*self.shape)

class Conv(Layer):
    def __init__(self, in_depth, out_depth, k, p, s):
        self.W = np.random.randn(out_depth, k**2*in_depth)*np.sqrt(2.0/(k**2*in_depth))
        self.in_depth = in_depth
        self.out_depth = out_depth
        self.k = k
        self.p = p
        self.s = s
        # self.updater = Momentum(0.9, self.W.shape)
        # self.updater = Updater()
        self.updater = Adam(0.9, 0.999, self.W.shape)

    def im2col(self, x):
        self.n = x.shape[0]
        self.in_size = np.array(x.shape[1:3])
        self.out_size = (self.in_size+2*self.p-self.k)//self.s+1

        arr = []
        n = x.shape[0]
        (h, w) = x.shape[1:3]

        z = np.zeros((n, h, self.p, self.in_depth))
        x = np.concatenate([z, x, z], axis=2)
        w+= 2*self.p
        z = np.zeros((n, self.p, w, self.in_depth))
        x = np.concatenate([z, x, z], axis=1)
        h+= 2*self.p

        for d in range(self.in_depth):
            for i in range(self.k):
                for j in range(self.k):
                    arr.append(x[:, i:h+1+i-self.k:self.s, j:w+1+j-self.k:self.s, d].reshape(-1))

        return np.stack(arr)
    def col2im(self, x):
        n = self.n
        y = np.zeros((n, self.in_size[0]+2*self.p, self.in_size[1]+2*self.p, self.in_depth))
        h, w = self.in_size+2*self.p
        for d in range(self.in_depth):
            for i in range(self.k):
                for j in range(self.k):
                    y[:, i:h+1+i-self.k:self.s, j:w+1+j-self.k:self.s, d]+= (x[d*self.k**2+i*self.k+j,:].reshape(n, self.out_size[0], self.out_size[1]))
        return y[:, self.p:h-self.p, self.p:w-self.p, :]
    def forward(self, x):
        self.x = x
        self.x_col = self.im2col(x)
        self.y_col = np.dot(self.W, self.x_col)
        self.y = np.moveaxis(self.y_col.reshape(self.out_depth, self.n, self.out_size[0], self.out_size[1]), 0, -1)
        return self.y
    def backward(self, dy):
        n = dy.shape[0]
        self.dy_col = np.moveaxis(dy, -1, 0).reshape(self.out_depth, -1)
        self.dW = np.dot(self.dy_col, self.x_col.T)/n
        self.dx_col = np.dot(self.W.T, self.dy_col)
        self.dx = self.col2im(self.dx_col)
        return self.dx
    def learn(self, learning_rate, weight_decay):
        self.W+= -learning_rate*self.updater.computeGradient(self.dW) -weight_decay*self.W

class NN:
    def __init__(self, layers):
        self.layers = layers
        self.loss = 0
        self.acc = 0
    def shuffle(self, x, y):
        n = x.shape[0]
        order = np.arange(n)
        np.random.shuffle(order)
        x[:] = x[order]
        y[:] = y[order]
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            # print("VAR->", np.var(x))
        return x
    def backward(self, y):
        dy = y
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
            # print("VAR", np.std(dy))
        # print()
    def learn(self):
        for layer in self.layers:
            layer.learn(self.learning_rate, self.weight_decay)
    def computeLossAndAcc(self, y, y_pred):
        loss = -np.log(y_pred[range(y.shape[0]), y]).sum()
        acc = np.count_nonzero((np.argmax(y_pred, axis=1) - y) == 0)
        return loss, acc
    def nextBatch(self, x, y):
        y_pred = self.forward(x)
        l, a = self.computeLossAndAcc(y, y_pred)
        self.loss+= l
        self.acc+= a
        self.backward(y)
        self.learn()
    def nextEpoch(self, x, y):
        self.loss, self.acc = 0, 0
        self.shuffle(x, y)
        n = x.shape[0]
        for i in range(n//self.batchSize):
            self.nextBatch(x[i:(i+self.batchSize)], y[i:(i+self.batchSize)])
        self.loss/= n
        self.acc/= n
    def validate(self, x, y):
        y_pred = self.forward(x)
        self.loss, self.acc = np.array(self.computeLossAndAcc(y, y_pred))/x.shape[0]
    def train(self, x, y, numberOfEpochs=100, batchSize=16, learning_rate=1e-3, weight_decay=1e-4):
        self.batchSize = batchSize
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.shuffle(x, y)
        n = x.shape[0]
        m = 4*n//5
        self.x_train = x[:m]
        self.y_train = y[:m]
        self.x_val = x[m:]
        self.y_val = y[m:]
        for i in range(numberOfEpochs):
            self.nextEpoch(self.x_train, self.y_train)
            print(i, self.loss, self.acc, end="\t")
            self.validate(self.x_val, self.y_val)
            print(self.loss, self.acc)

if __name__ == "__main__":
    print("NN module")
