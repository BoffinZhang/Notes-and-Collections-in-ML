import mnist_loader
from Network import Network


print('Loading data')
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# 构建network
print('Building network')
net = Network([784,30,10])

print('Running SGD algorithm')
# 进行随机梯度下降算法，30个epoch，每个mini_batch大小为10，步长为3.0
net.SGD(training_data,30,10,3.0,test_data = test_data)
