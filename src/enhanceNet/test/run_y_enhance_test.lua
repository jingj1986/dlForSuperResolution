require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'cudnn'
require 'utils.InstanceNormalization'
require 'utils.TotalVariation'
local utils = require 'utils.utils'

util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
  dataset = 'folder', 
  batchSize=1,
  niter=1,
  ntrain = math.huge, 
  gpu=2,
  backend = 'cuda',
  nThreads = 1,
  scale=1,
  chan = 3,
  loadSize=50,

  wild=352,
  high=288,


--[[
  wild=512,
  high=384,
]]--
  t_folder='./test_pic/',
  --model_file='./model/test_model_G',
  model_file='./model/test_model_t7',
  result_path='./result/'
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local DataLoader = paths.dofile('data/data.lua')
data = DataLoader.new(opt.nThreads,  opt)

local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)

--local modelG = util.load(opt.model_file, opt.gpu)
local modelG = torch.load(opt.model_file):type(dtype)
--local modelG = torch.load(opt.model_file).model:type(dtype)
cutorch.setDevice(opt.gpu)
modelG = modelG:cuda()
local parametersG, gradientsG = modelG:getParameters()

cnt=1
local vgg_mean = {103.939, 116.779, 123.68}
function deprocess(img)
--    local mean = img.new(vgg_mean):view(3, 1, 1):expandAs(img)
--    local perm = torch.LongTensor{3, 2, 1}
--    return (img + mean):div(255):index(1, perm)
    return img:add(1):div(2)
end
--modelG:zeroGradParameters()

--function y2rgb(img)  
--end

for i = 1, opt.niter do
  real_uncropped,input= data:getBatch()
  --real=real_uncropped[{{},{},{1,1+93-1},{1,1+93-1}}]
  --real=real_uncropped[{{},{},{1,1+512-1},{1,1+512-1}}]
  --real=real_uncropped[{{},{},{1,opt.loadSize},{1,opt.loadSize}}]
  real=real_uncropped[{{},{},{1,opt.high},{1,opt.wild}}]
  --input = input:cuda()
  --test_input = {1,1,opt.high,opt.wild}
  test_input = torch.Tensor(1,1,opt.high,opt.wild)
  test_input[1] = image.rgb2y(input[1])
    
  test_input = test_input:cuda()
  fake = modelG:forward(test_input)
  print("END of forward")
  for j=1,opt.batchSize do
    
    org = deprocess(input[1])
    org = image.rgb2yuv(org)
    org = image.scale(org, opt.wild*2, opt.high*2, 'bilinear')
--[[    
    fake_rgb = torch.Tensor(3, opt.high*2,opt.wild*2)
    fake_rgb[1]:copy(deprocess(fake[j][1]))
    fake_rgb[2]:copy(org[2])
    fake_rgb[3]:copy(org[3])
    fake_rgb = image.yuv2rgb(fake_rgb)
    fake_rgb[fake_rgb:gt(1)]=1
    fake_rgb[fake_rgb:lt(0)]=0
]]--
    image.save(string.format('%s/raw_%04d.png',opt.result_path,cnt),deprocess(real[j]))
--    image.save(string.format('%s/fake_%04d.png',opt.result_path,cnt),image.toDisplayTensor(fake_rgb))
    image.save(string.format('%s/fake_%04d.png',opt.result_path,cnt),deprocess(fake[j][1]))
    cnt=cnt+1
  end
end




