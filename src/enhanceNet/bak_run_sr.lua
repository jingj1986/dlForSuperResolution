require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'weight-init'

require 'utils.PerceptualCriterion'
require 'utils.TotalVariation'
require 'utils.InstanceNormalization' 

require 'tv.SpatialSimpleGradFilter'
require 'tv.SpatialTVNorm'
require 'tv.SpatialTVNormCriterion'

local utils = require 'utils.utils'
--local G = require 'enhanceNet.lua'
util = paths.dofile('util.lua')

opt = {
  dataset = 'folder',
  lr = 0.0001,
  beta1 = 0.9,  
  batchSize=16,
  niter=10,
  loadSize=96,
  ntrain = math.huge, 
  name='super_resolution',
  gpu=3,
  nThreads = 4,
  scale=4,
  --t_folder = './test',
  t_folder='/data/images_lib/imagenet/ILSVRC2012/ILSVRC2012_img_train/',
  model_folder='./model/',
}

local loss_net = torch.load('models/VGG16.t7')
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
print("Dataset: " .. opt.dataset, " Size: ", data:size())

local real_label=1
local fake_label=0
--local G=require 'bak_enhanceNet'
--local G=require 'bak.lua'
--local G = require 'adversarial_G.lua'

local G=require 'enhanceNet'
local modelG = require('weight-init')(G(), 'kaiming')
--local modelG = torch.load('./model/fast_content').model:type(dtype)

local D=require 'adversarial_D.lua'
local modelD = require('weight-init')(D(),'kaiming')
local criterion = nn.BCECriterion() 
local criterion_tv = nn.SpatialTVNormCriterion()
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

local input = torch.Tensor(opt.batchSize, 3, opt.loadSize/opt.scale, opt.loadSize/opt.scale) 
local real = torch.Tensor(opt.batchSize,3,opt.loadSize,opt.loadSize)
local real_uncropped = torch.Tensor(opt.batchSize,3,opt.loadSize,opt.loadSize)
local output = torch.Tensor(opt.batchSize,3,opt.loadSize,opt.loadSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local test = torch.Tensor(3, opt.loadSize, opt.loadSize) 
local test2 = torch.Tensor(3, opt.loadSize/opt.scale, opt.loadSize/opt.scale) 
local label = torch.Tensor(opt.batchSize)

if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  
   real=real:cuda(); 
   output=output:cuda();
   modelG=modelG:cuda()
   modelD=modelD:cuda()
   criterion:cuda()
   criterion_tv:cuda();
   criterion_mse:cuda();
   criterion_percep:cuda() 
   label=label:cuda()
end

local parametersG, gradientsG = modelG:getParameters()
local parametersD,gradientsD= modelD:getParameters()
local fDx=function(x)
    if x ~= parametersD then
          parametersD:copy(x)
    end
    modelD:zeroGradParameters()

    real_uncropped,input= data:getBatch()
    real_uncropped=real_uncropped:cuda()
    real=real_uncropped[{{},{},{1,96},{1,96}}]
    real=real:cuda()

    label:fill(real_label)
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

    --D
    errG_D = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    local df_dg=modelD:updateGradInput(fake,df_do)

    --MSE
--    errG_mse=criterion_mse:forward(fake,real)
--    local df_do_mse=criterion_mse:backward(fake,real)

    --VGG
    errG_VGG = criterion_percep:forward(fake,real)
    local df_do_vgg = criterion_percep:backward(fake, real)

    --TV
    local target = nil
    errG_TV = criterion_tv:forward(fake, target)
    print(errG_TV)
    local df_do_tv = criterion_tv:backward(fake,target)

    modelG:backward(input, df_dg + 0.2*df_do_vgg + df_do_tv)
    err_all=errG_D + 0.2*errG_VGG + errG_TV
--    modelG:backward(input,50*df_dg+df_do_vgg)
--    err_all=50*errG_D+errG_VGG

    return err_all,gradientsG
end
local fGx_only = function(x)
    modelG:zeroGradParameters()

    real_uncropped,input= data:getBatch()
    real_uncropped=real_uncropped:cuda()
    real=real_uncropped[{{},{},{1,opt.loadSize},{1,opt.loadSize}}]
    real=real:cuda()
    input = input:cuda()
    fake = modelG:forward(input)
    
    --MSE
    --local errG = criterion_mse:forward(fake, real)
    --local df_do_mse = criterion_mse:backward(fake, real)

    --VGG
    errG_VGG = criterion_percep:forward(fake,real)
    local df_do_vgg = criterion_percep:backward(fake, real)

    modelG:backward(input, df_do_vgg)
    err_all = errG_VGG

    return err_all, gradientsG 
end
local vgg_mean = {103.939, 116.779, 123.68}

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
end

local counter = 0
for epoch = 1, opt.niter do
   epoch_tm:reset()
   
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()

      --optim.adam(fDx, parametersD, optimStateD)
      --optim.adam(fGx, parametersG, optimStateG)
      optim.adam(fGx_only, parametersG, optimStateG)

      counter = counter + 1
      if counter % 1000 == 0 then
          test:copy(real[1])
          local real_rgb=test
          image.save("./result/" .. opt.name..counter..'_real.png', deprocess(real_rgb))
          test2:copy(input[1])
          image.save("./result/" .. opt.name..counter..'_input.png',deprocess(test2))
          fake = deprocess(fake[1])
          fake[fake:gt(1)]=1
          fake[fake:lt(0)]=0
          --test:copy(fake[1])
          image.save("./result/" .. opt.name..counter..'_fake.png',fake)
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
   --parametersG, gradientsG = nil, nil
   if epoch % 1 == 0 then
     util.save(opt.model_folder .. opt.name .. '_adversarial_G_' .. epoch, modelG, opt.gpu)
     util.save(opt.model_folder .. opt.name .. '_adversarial_D_' .. epoch, modelD, opt.gpu)
   end
   if epoch == 2 then
        optimStateG.learningRate = optimStateG.learningRate * 0.1
   end

   if epoch % 6 == 0 then
        optimStateG.learningRate = optimStateG.learningRate * 0.1
   end
   parametersG, gradientsG = modelG:getParameters()
   parametersD, gradientsD=modelD:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
