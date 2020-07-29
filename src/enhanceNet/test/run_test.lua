require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'cudnn'
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
  nThreads = 2,
  scale=4,
  --scale=1,
  chan = 3,
  loadSize=128,
  wild = 512,
  high = 512,
  t_folder='./test_pic/',
  model_file='./model/test_model_t7',
  result_path='./result/'
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

local DataLoader = paths.dofile('data/data.lua')
data = DataLoader.new(opt.nThreads,  opt)

local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
local modelG=util.load(opt.model_file,opt.gpu)
print(modelG)
cutorch.setDevice(opt.gpu)
modelG = modelG:cuda()
--local parametersG, gradientsG = modelG:getParameters()

cnt=1
--modelG:zeroGradParameters()
local vgg_mean = {103.939, 116.779, 123.68}
function deprocess(img)
    local mean = img.new(vgg_mean):view(3, 1, 1):expandAs(img)
    local perm = torch.LongTensor{3, 2, 1}
    return (img + mean):div(255):index(1, perm)
end
for i = 1, opt.niter do
  real_uncropped,input= data:getBatch()
  --real=real_uncropped[{{},{},{1,1+93-1},{1,1+93-1}}]
  --real=real_uncropped[{{},{},{1,1+512-1},{1,1+512-1}}]
  --real=real_uncropped[{{},{},{1,opt.loadSize},{1,opt.loadSize}}]
  print(input:size())
  real=real_uncropped[{{},{},{1,opt.high},{1,opt.wild}}]
  print(real:size())
  --input = input:mul(2):add(-1)
  input = input:cuda()
  print(i)
  fake = modelG:forward(input)
  print("END of forward")
--  fake[fake:gt(1)]=1
--  fake[fake:lt(0)]=0
  for j=1,opt.batchSize do
--    image.save(string.format('%s/raw_%04d.png',opt.result_path,cnt),image.toDisplayTensor(real[j]))
--    image.save(string.format('%s/fake_%04d.png',opt.result_path,cnt),image.toDisplayTensor(fake[j]))
    image.save(string.format('%s/raw_%04d.png',opt.result_path,cnt),deprocess(real[j]))
    image.save(string.format('%s/fake_%04d.png',opt.result_path,cnt),deprocess(fake[j]))
    cnt=cnt+1
  end
end




