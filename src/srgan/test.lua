require 'torch'
require 'nn'

model = torch.load('models/vgg16.t7')
print(model)
