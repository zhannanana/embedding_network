import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

def test_grad():
    lr = 0.01
    epoches = 1000
    x = torch.randn(1, 1, requires_grad=True)  # 随机的初始值,变量必须是torch的变量呀
    print(x.grad, x.data)   # grad梯度，和值，x.grad.data在这个点求导的值
    for epoch in range(epoches):
        y = x**2 + 2*x + 1   #每次都要计算一遍
        y.backward()   #自动求一次导，但是x不会自己去变更？
        print("grad", x.grad.data, "x.data", x.data)  # x的梯度值
        x.data = x.data - lr*x.grad.data
        x.grad.data.zero_()   # 为什么每次梯度要清空，不清空就还是接着上次的，对于复杂的张量如何做呢？

    print(x.data)


class TestGrad(object):
    def __init__(self):
        self.x = torch.randn(1, 20, requires_grad=True)
        self.x2 = torch.randn(1, 20, requires_grad=True)

    def gong1(self):
        t = self.x*self.x
        return t

    def gong2(self):
        n = self.x*2+1
        return n

    def gong3(self, lr):
        y = torch.add(self.gong1(), self.gong2())
        y.backward()
        print("grad", self.x.grad, "x.data", self.x)  # x的梯度值
        # print("before:", self.x)
        self.x.data = torch.add(self.x.data, -lr*self.x.grad.data)
        self.x.grad.data.zero_()
        # print("after:", self.x)
        return y

"""
testc = TestGrad()
lr = 0.01
echopes = 1000
print("初始值：", testc.x)
print("testc x", testc.x.grad, testc.x.data)
for i in range(echopes):
    testc.gong3(lr)  # 先计算一遍 然后求导
"""

# 跟我想的一样，把它放到参数列表里，就能通过optimizer.step()更新参数了
x1 = torch.randn(1, 1, requires_grad=True)
#x2 = torch.randn(1, 1, requires_grad=True)
optimizer = optim.SGD([x1], lr=0.01, momentum=0.9)
optimizer.zero_grad()
for i in range(1000):
    loss = x1 * x1 + 2 * x1 + 1
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("loss:", loss, "x:", x1)


