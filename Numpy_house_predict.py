
from Tensor_3 import load_data
import numpy as np
import matplotlib.pyplot as plt

class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost

    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)

        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epochs):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                # print(self.w.shape)
                # print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                      format(epoch_id, iter_id, loss))

        return losses

train_data, test_data = load_data()
# 创建网络

net = Network(13)
# 启动训练
losses = net.train(train_data, num_epochs=50, batch_size=100, eta=0.1)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()











'''
ndim_1_tensor = paddle.to_tensor([1mpy()
ndim_2_tensor = numpy.array([1, 2, 3])

print(ndim_1_tensor)
print('.........................')
pass

print(paddle.linspace(2, 9, 4))


paddle.zeros([m, n])             # 创建数据全为0，shape为[m, n]的Tensor
paddle.ones([m, n])              # 创建数据全为1，shape为[m, n]的Tensor
paddle.full([m, n], 10)          # 创建数据全为10，shape为[m, n]的Tensor
paddle.arange(start, end, step)  # 创建从start到end，步长为step的Tensor
paddle.linspace(start, end, num) # 创建从start到end，元素个数固定为num的Tensor
'''
'''
a = np.array([10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
a_num = len(a)


print(a.shape)
print(a.shape[0])

print('...........................')
a = a.reshape(a.shape[0]//7, 7)


print(a.shape)
print(a.shape[0])
print(a.shape[1])

print('...........................')

a = a.reshape(2, 7, 1)
print(a.shape)
print(a.shape[0])
print(a.shape[1])
print(a.shape[2])
print(a)
print(',,,,,,,,,,,,,,,,,,,,,')
print(a[1][2][0])


a = a.reshape(2, 7)
print(a[0].shape)
print(a)
print('..................')
print(a.max(axis=0), a.min(axis=1))
print(a[:, 1])
'''