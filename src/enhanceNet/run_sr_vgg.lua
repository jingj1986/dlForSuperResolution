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
  batchSize=64,
  niter= 200,
  loadSize=96,
  ntrain = math.huge, 
  name='sr',
  gpu=3,
  nThreads = 16,
  scale=4,
  chan = 3,
  --t_folder = './test',
  --t_folder='/data/images_lib/imagenet/ILSVRC2012/ILSVRC2012_img_train/',
--  t_folder = '/data/images_lib/video/sidalin/high_a/',
--  t_folder = '/data/images_lib/xiju/x2_768/high/',
--  t_folder = '/data/images_lib/xiju/frames/x2-xjall/high/',
--  t_folder = '/data/images_lib/xiju/x2_768_sm/high/',
--  t_folder = '/data/images_lib/x2/high/',
--  t_folder = '/data/images_lib/xiju/x2_rm/high/',
--  t_folder = '/data/images_lib/xiju/frames/org/',
  t_folder = '/data/images_lib/xiju/x4/high/',
  --model_folder='./model/',
  backend = 'cuda',
  use_cudnn = 1,
}

local loss_net = torch.load('models/VGG19.t7')
local crit_args = {
    cnn = loss_net,
    content_layers = {'10', '37'}, --2,2
--    content_layers = {'9', '36'}, --2,2
    content_weights = {0.2, 0.02},
}

torch.manualSeed(1)
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
local DataLoader = paths.dofile('data/data.lua')
data = DataLoader.new(opt.nThreads, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

local real_label=1
local fake_label=0

local G=require 'enhanceNet'
local modelG = require('weight-init')(G(), 'kaiming')
--local modelG = torch.load('./model/bak/xjx2_mse_pretrain.t7'):type(dtype)
print(modelG)

local D=require 'adversarial_D.lua'
local modelD = require('weight-init')(D(),'kaiming')
--local modelD = torch.load('./model/xjx2_vggmse_600_D.t7'):type(dtype)

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
   modelD=modelD:cuda()
   criterion:cuda()
   criterion_tv:cuda();
   criterion_mse:cuda();
   criterion_percep:cuda() 
   label=label:cuda()
end

if use_cudnn then
    cudnn.convert(modelG, cudnn)
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
--[[
local vgg_weight=0.999
local d_weight = 0.001
]]--

local vgg_weight=1
local d_weight=1


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
    errG_mse=criterion_mse:forward(fake,real)
    local df_do_mse=criterion_mse:backward(fake,real)

    --VGG
    errG_VGG = criterion_percep:forward(fake,real)
    local df_do_vgg = criterion_percep:backward(fake, real)

    --TV
--    local target = nil
--    errG_TV = criterion_tv:forward(fake, target)
--    print(errG_TV)
--    local df_do_tv = criterion_tv:backward(fake,target)
--    print("D:" .. errG_D .. ", MSE:".. errG_mse .. ", VGG:" .. errG_VGG)
    --modelG:backward(input, df_dg*0.1 + df_do_vgg*0.01 + df_do_mse)

    modelG:backward(input, df_dg*d_weight + df_do_vgg*vgg_weight + df_do_mse)
    err_all=errG_D*d_weight + errG_VGG*vgg_weight + errG_mse
    print("D: " .. d_weight*errG_D .. ", VGG: " .. errG_VGG*vgg_weight .. ", MSE: " .. errG_mse)

--[[
    modelG:backward(input, df_dg*d_weight + df_do_vgg*vgg_weight)
    err_all=errG_D*d_weight + errG_VGG*vgg_weight
]]--

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
    local errG_mse = criterion_mse:forward(fake, real)
    local df_do_mse = criterion_mse:backward(fake, real)

    --VGG
    --errG_VGG = criterion_percep:forward(fake,real)
    --local df_do_vgg = criterion_percep:backward(fake, real)

    --modelG:backward(input, df_do_vgg)
    --err_all = errG_VGG
    modelG:backward(input, df_do_mse)
    err_all = errG_mse

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
--  check_input(img)
  local mean = img.new(vgg_mean):view(3, 1, 1):expandAs(img)
  local perm = torch.LongTensor{3, 2, 1}
  return (img + mean):div(255):index(1, perm)
--    return img:add(1):div(2)
end

local counter = 0
for epoch = 1, opt.niter do
   epoch_tm:reset()
   
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()

      optim.adam(fDx, parametersD, optimStateD)
      optim.adam(fGx, parametersG, optimStateG)
--      optim.adam(fGx_only, parametersG, optimStateG)

      counter = counter + 1
      if counter % 2000 == 0 then
          test:copy(real[1])
          local real_rgb=test
          image.save("./result/" .. opt.name..counter..'_real.png', deprocess(real_rgb))
          test2:copy(input[1])
          image.save("./result/" .. opt.name..counter..'_input.png',deprocess(test2))
          fake = deprocess(fake[1])
--          fake[fake:gt(1)]=1
--          fake[fake:lt(0)]=0
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
   parametersG, gradientsG = nil, nil
   if epoch % 10 == 0 then
--     util.save('./model/xjx2_vgg_' .. epoch .. '_G.t7', modelG, opt.gpu)
--     util.save('./model/xjx2_vgg_' .. epoch .. '_D.t7', modelD, opt.gpu)
    torch.save('./model/x4_vggmseD_' .. epoch .. '_G.t7', modelG)
    torch.save('./model/x4_vggmseD_' .. epoch .. '_D.t7', modelD)
--        modelG:clearState()
--        modelG:float()
--        if use_cudnn then cudnn.convert(modelG, nn) end
--        torch.save('./model/xjx2_vgg_' .. epoch .. '_G.t7', modelG:clearState() )
--        torch.save('./model/xjx2_vgg_' .. epoch .. '_D.t7', modelD:clearState() )
--        if use_cudnn then cudnn.convert(modelG, cudnn) end
--        modelG:type(dtype)
   end
   if epoch == 150 then
        optimStateG.learningRate = optimStateG.learningRate * 0.1
   end
   if epoch == 200 then
        if use_cudnn then cudnn.convert(modelG, nn) end
        torch.save('./model/x4_vggmseD_s' .. epoch .. '_G.t7', modelG:clearState() )
        torch.save('./model/x4_vggmseD_s' .. epoch .. '_D.t7', modelD:clearState() )
        if use_cudnn then cudnn.convert(modelG, cudnn) end
  end

--[[
   if epoch % 150 == 0 then
        optimStateG.learningRate = optimStateG.learningRate * 0.2
--        vgg_weight = vgg_weight * 1.01
   end
]]--
   parametersG, gradientsG = modelG:getParameters()
   parametersD, gradientsD = modelD:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
