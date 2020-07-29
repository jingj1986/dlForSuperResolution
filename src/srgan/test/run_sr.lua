require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'weight-init'
require 'utils.PerceptualCriterion'
local G=require 'adversarial_G.lua'
local D=require 'adversarial_D.lua'

local utils = require 'utils.utils'
local preprocess = require 'utils.preprocess'
util = paths.dofile('util.lua')

opt = {
  dataset = 'folder',
  lr = 0.0001,
  beta1 = 0.9,  
  batchSize=16,
  niter=200,
  loadSize=96,
  ntrain = math.huge, 
  name='super_resolution',
  gpu=1,
  chan=1,
  nThreads = 4,
  scale=4,
  use_cudnn = 1,
  backend = 'cuda',
  --t_folder='/data/images_lib/imagenet/ILSVRC2012/ILSVRC2012_img_train/',
  t_folder = './test',
  model_folder='./model/',
}

local loss_net = torch.load('models/vgg16.t7')
--print(loss_net)
local crit_args = {
    cnn = loss_net,
    content_layers = {"16"},
    content_weights = {1.0},
}

torch.manualSeed(1)

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)

local DataLoader = paths.dofile('data/data.lua')
data = DataLoader.new(opt.nThreads, opt)

local real_label=1
local fake_label=0

local G=require 'adversarial_G.lua'
local modelG = require('weight-init')(G(), 'kaiming')
--local modelG = util.load('./model/pre_mse_G_8', opt.gpu)
local D=require 'adversarial_D.lua'
local modelD = require('weight-init')(D(),'kaiming')
local criterion = nn.BCECriterion() 
local criterion_mse = nn.MSECriterion()
local criterion_percep = nn.PerceptualCriterion(crit_args):type(dtype)

optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
   weightDecay=0.0001,
}
optimStateD = {
   learningRate = opt.lr*0.1,
   beta1 = opt.beta1,
   weightDecay=0.0001,
}

local input = torch.Tensor(opt.batchSize, 3, opt.loadSize/4, opt.loadSize/4) 
--local real = torch.Tensor(opt.batchSize,3,opt.loadSize-3,opt.loadSize-3)
local real = torch.Tensor(opt.batchSize,3,opt.loadSize,opt.loadSize)
local real_uncropped = torch.Tensor(opt.batchSize,3,opt.loadSize,opt.loadSize)
--local output = torch.Tensor(opt.batchSize,3,opt.loadSize-3,opt.loadSize-3)
local output = torch.Tensor(opt.batchSize,3,opt.loadSize,opt.loadSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
--local test = torch.Tensor(3, opt.loadSize-3, opt.loadSize-3) 
local test = torch.Tensor(3, opt.loadSize, opt.loadSize) 
local test2 = torch.Tensor(3, opt.loadSize/4, opt.loadSize/4) 
local label = torch.Tensor(opt.batchSize)

if opt.gpu > 0 then
   require 'cunn'
   print('cunn used')
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  
   real=real:cuda(); 
   output=output:cuda();
   modelG=modelG:cuda()
   modelD=modelD:cuda()
   criterion:cuda()
   criterion_mse:cuda(); 
   label=label:cuda()
end

local parametersG, gradientsG = modelG:getParameters()
local parametersD,gradientsD= modelD:getParameters()

local fDx=function(x)
    if x ~= parametersD then
          parametersD:copy(x)
    end
    real_uncropped,input= data:getBatch()
    real_uncropped=real_uncropped:cuda()
    --real=real_uncropped[{{},{},{1,1+93-1},{1,1+93-1}}]
    real=real_uncropped[{{},{},{1,96},{1,96}}]
    real=real:cuda()

    local output=modelD:forward(real)
    local errD_real=criterion:forward(output,label)
    local df_do = criterion:backward(output, label)
    modelD:backward(real,df_do)

    input=input:cuda()
    fake = modelG:forward(input)
    label:fill(fake_label)
    local output=modelD:forward(fake)
    local errD_fake=criterion:forward(output,label)
    local df_do = criterion:backward(output, label)
    modelD:backward(fake, df_do)

    errD = errD_real + errD_fake
    return errD, gradientsD
end

local fGx=function(x)
    modelG:zeroGradParameters()
    label:fill(real_label)
    local output=modelD.output
    
    input=input:cuda()
    real=real:cuda()
    errG = criterion:forward(output, label)
    --errG_mse=criterion_mse:forward(fake,real)
    fake_pr = preprocess['vgg'].preprocess(fake)
    real_pr = preprocess['vgg'].preprocess(real)
    errG_per = criterion_percep:forward(fake_pr,real_pr)
    --errG_per = criterion_percep:forward(fake,real)

    local df_do = criterion:backward(output, label)
    --local df_do_mse=criterion_mse:backward(fake,real)
    local df_do_per = criterion_percep:backward(fake_pr, real_pr)
    --local df_do_per = criterion_percep:backward(fake, real)
    --local df_do_per=df_do_per:div(255):index(2, torch.LongTensor{3,2,1})
    local df_do_per=df_do_per:index(2, torch.LongTensor{3,2,1})

    local df_dg=modelD:updateGradInput(fake,df_do)
    --modelG:backward(input,0.001*df_dg+0.999*df_do_mse)
    --err_all=0.001*errG+0.999*errG_mse
    modelG:backward(input,0.001*df_dg+0.999*df_do_per)
    --print(df_dg:size())
    --print(df_do_per:size())
    err_all=0.001*errG+0.999*errG_per
    return err_all,gradientsG
end

local fGx_msevgg = function(x)
    modelG:zeroGradParameters()
    real_uncropped,input= data:getBatch()
    real = real_uncropped:cuda()
    input = input:cuda()
    
    fake = modelG:forward(input)
    local errG_mse = criterion_mse:forward(fake, real) 
    local df_do_mse = criterion_mse:backward(fake, real)

    local fake_pr = preprocess['vgg'].preprocess(fake)
    local real_pr = preprocess['vgg'].preprocess(real)

    local errG_per = criterion_percep:forward(fake_pr,real_pr)
    local df_do_per = criterion_percep:backward(fake_pr, real_pr)

    df_do_per = df_do_per:index(2, torch.LongTensor{3,2,1})
    modelG:backward(input, 0.5 * df_do_per + 0.5 * df_do_mse)
    err_all = 0.5 * errG_per + 0.5 * errG_mse
    
    return err_all, gradientsG
end

local fGx_only = function(x)
    modelG:zeroGradParameters()
    real_uncropped,input= data:getBatch()
    real = real_uncropped:cuda()
    input = input:cuda()
    
    fake = modelG:forward(input)

    local fake_pr = preprocess['vgg'].preprocess(fake)
    local real_pr = preprocess['vgg'].preprocess(real)

    local errG_per = criterion_percep:forward(fake_pr,real_pr)
    local df_do_per = criterion_percep:backward(fake_pr, real_pr)

    df_do_per = df_do_per:index(2, torch.LongTensor{3,2,1})
    modelG:backward(input, df_do_per)
    err_all = errG_per
    
    return err_all, gradientsG
end

local counter = 0
for epoch = 1, opt.niter do
   epoch_tm:reset()
   
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()

      optim.adam(fDx, parametersD, optimStateD)
      optim.adam(fGx, parametersG, optimStateG)
      --optim.adam(fGx_msevgg, parametersG, optimStateG)

      counter = counter + 1
      print("count: " .. counter)
      if counter % 1000 == 0 then
          test:copy(real[1])
          local real_rgb=test
          image.save("./result/" .. opt.name..counter..'_real.png',real_rgb)
          test2:copy(input[1])
          image.save("./result/" .. opt.name..counter..'_input.png',test2)
          fake[fake:gt(1)]=1
          fake[fake:lt(0)]=0
          image.save("./result/" .. opt.name..counter..'_fake.png',fake[1])
      end
      
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f '
                   .. '  Err_G: %.4f Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, 
                 err_all and err_all or -1, errD and errD or -1))
      end
   end
   --paths.mkdir('/media/DATA/MODELS/SUPER_RES/checkpoints')
   parametersD, gradientsD= nil, nil
   parametersG, gradientsG = nil, nil
   if epoch % 10000 == 0 then
     util.save(opt.model_folder .. opt.name .. '_adversarial_G_' .. epoch, modelG, opt.gpu)
     util.save(opt.model_folder .. opt.name .. '_adversarial_D_' .. epoch, modelD, opt.gpu)
   end
   parametersG, gradientsG = modelG:getParameters()
   parametersD, gradientsD=modelD:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end




