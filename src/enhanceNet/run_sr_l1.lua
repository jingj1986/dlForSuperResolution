require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'weight-init'

local utils = require 'utils.utils'
util = paths.dofile('util.lua')

opt = {
  dataset = 'folder',
  lr = 0.0001,
  beta1 = 0.9,  
  batchSize=14,
  niter=10000,
  loadSize=96,
  ntrain = math.huge, 
  name='super_resolution',
  gpu=1,
  nThreads = 4,
  scale=2,
  chan = 3,
  t_folder = '/home/user/data/DIV2K/train/high/',
  backend = 'cuda',
  use_cudnn = 1,
}

--torch.manualSeed(1)
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
local DataLoader = paths.dofile('data/data.lua')
data = DataLoader.new(opt.nThreads, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

local real_label=1
local fake_label=0

local G=require '256_G.lua'
local modelG = require('weight-init')(G(), 'kaiming')
--local modelG = G()
--local modelG = torch.load('./model/EDSR_4000.t7'):type(dtype)

--==== ADD submean
local meanVec = torch.Tensor({0.4488, 0.4371, 0.4040}):mul(255)
local subMean = nn.SpatialConvolution(3, 3, 1, 1)
subMean.weight:copy(torch.eye(3, 3):reshape(3, 3, 1, 1))
subMean.bias:copy(torch.mul(meanVec, -1))
local addMean = nn.SpatialConvolution(3, 3, 1, 1)
addMean.weight:copy(torch.eye(3,3):reshape(3, 3, 1, 1))
addMean.bias:copy(meanVec)
modelG:insert(subMean, 1)
modelG:insert(addMean)

--print(modelG)

--local criterion = nn.BCECriterion() 
local criterion = nn.MultiCriterion()
local criterion_mse = nn.MSECriterion()
local criterion_abs = nn.AbsCriterion()
criterion_abs.sizeAverage = true
criterion:add(criterion_abs, 1)

optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
   weightDecay=0.0001,
}

local input = torch.Tensor(opt.batchSize, opt.chan, opt.loadSize/opt.scale, opt.loadSize/opt.scale) 
local real = torch.Tensor(opt.batchSize, opt.chan,opt.loadSize,opt.loadSize)
local real_uncropped = torch.Tensor(opt.batchSize,opt.chan,opt.loadSize,opt.loadSize)
local output = torch.Tensor(opt.batchSize,opt.chan,opt.loadSize,opt.loadSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local test = torch.Tensor(opt.chan, opt.loadSize, opt.loadSize) 
local test2 = torch.Tensor(opt.chan, opt.loadSize/opt.scale, opt.loadSize/opt.scale) 
local label = torch.Tensor(opt.batchSize)

if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  
   real=real:cuda(); 
   output=output:cuda();
   modelG=modelG:cuda()
   criterion:cuda()
   criterion_mse:cuda();
--   criterion_abs:cuda();
   label=label:cuda()
end

if use_cudnn then
    cudnn.convert(modelG, cudnn)
end

local parametersG, gradientsG = modelG:getParameters()
local fGx_only = function(x)
    modelG:zeroGradParameters()

    real_uncropped,input= data:getBatch()
    real_uncropped=real_uncropped:cuda()
    real=real_uncropped[{{},{},{1,opt.loadSize},{1,opt.loadSize}}]
    real=real:cuda()
    input = input:cuda()
    fake = modelG:forward(input)
    
    --MSE
    --local errG_mse = criterion_mse:forward(fake, real)
    --local df_do_mse = criterion_mse:backward(fake, real)

    --Abs L1
    --local errG_abs = criterion_abs:forward(fake, real)
    --local df_do_abs = criterion_abs:backward(fake, real)
    local errG_abs = criterion:forward(modelG.output, real)
    local df_do_abs = criterion:backward(fake, real)

    modelG:backward(input, df_do_abs)
    err_all = errG_abs

    return err_all, gradientsG 
end
local vgg_mean = {103.939, 116.779, 123.68}

--[[
function preprocess(img)
  check_input(img)
  local mean = img.new(vgg_mean):view(1, 3, 1, 1):expandAs(img)
  local perm = torch.LongTensor{3, 2, 1}
  return img:index(2, perm):mul(255):add(-1, mean)
end

local function check_input(img)
    assert(img:dim() == 3, 'img must be C x H x W')
    assert(img:size(1) == 3, 'img must have three channels')
end
-- Undo VGG preprocessing
function deprocess(img)
  check_input(img)
  local mean = img.new(vgg_mean):view(3, 1, 1):expandAs(img)
  local perm = torch.LongTensor{3, 2, 1}
  return (img + mean):div(255):index(1, perm)
--    return img:add(1):div(2)
end
]]--
local counter = 0
for epoch = 1, opt.niter do
   epoch_tm:reset()
   
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      optim.adam(fGx_only, parametersG, optimStateG)

      counter = counter + 1
      if counter % 1000 == 0 then
          test:copy(real[1])
          local real_rgb=test
          image.save("./result/" .. opt.name..counter..'_real.png', real_rgb:div(255))
          test2:copy(input[1])
          image.save("./result/" .. opt.name..counter..'_input.png',test2:div(255))
          test:copy(fake[1])
          image.save("./result/" .. opt.name..counter..'_fake.png',test:div(255))
      end
      
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f '
                   .. '  Err_G: %.4f '):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, 
                 err_all and err_all or -1))
      end
   end
   parametersD, gradientsD= nil, nil
   parametersG, gradientsG = nil, nil
   if epoch % 100 == 0 then
        torch.save('./model/EDSR_' .. epoch .. '.t7', modelG:clearState() )
   end
   if epoch % 3000 == 0 then
        optimStateG.learningRate = optimStateG.learningRate * 0.5
   end

   parametersG, gradientsG = modelG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
