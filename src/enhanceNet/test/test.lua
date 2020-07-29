require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'weight-init'
require 'utils.TotalVariation'
util = paths.dofile('util.lua')
--require 'utils.ShaveImage'
--[[
local G = require 'adversarial_G.lua'
local net = G()
--local net = require('weight-init')(G(), 'kaiming')
]]--

--local net = util.load('/home/user/project/waifu2x/waifu2x-master/models/photo/scale2.0x_model.t7', 3)
--local net = util.load('../vdsr_vgg/models/VDSR.t7', 3)
--local net = util.load('../fast-neural-style/models/eccv16/starry_night.t7', 3)
--local net = util.load('./model/super_resolution_adversarial_G_1', 2)
local net = util.load('./model/test_model_t7', 2)
--local net = util.load('../srgan_vgg/model/super_resolution_adversarial_G_10', 2)
--local net = torch.load('/home/user/project/model/resnet-18.t7')

print(net)
--local modules = net.modeules

--local modelG = require('weight-init')(G(), 'kaiming')
function loop_m(net)
    for i = 1, #net.modules do
        local m = net.modules[i]
        --print(i)
        print(m.__typename)
        if m.__typename == 'cudnn.SpatialConvolution' or m.__typename == 'nn.SpatialConvolution' then
            --print("NET " .. i)
            print(m.weight[{{},1}])
            --print(m.gradWeight[{{1,5},1}])
        elseif m.__typename == 'nn.Sequential' or m.__typename == 'nn.ConcatTable' then
            loop_m(m) 
        end
    end
end
--[[
for i = 1, #net.modules do
    local m = net.modules[i]
    print(m.__typename)
    if m.__typename == 'nn.SpatialConvolution' then
        print("NET " .. i)
        print(m.weight[{{1,5},1}])
    elseif m.__typename == 'nn.Sequential' then
        for j = 1, 
    end
end
]]--

loop_m(net)

--[[
layer=net.modules[1]
print(layer.__typename)
net.modules[1].weight = layer.weight:mul(0.01)
print(layer.weight:size())
print(layer.bias:size())
print(layer.gradWeight:size())
print(layer.weight)
]]--
