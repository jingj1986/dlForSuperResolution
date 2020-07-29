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
--  wild=110,
--  high=110,
  wild=352,
  high=288,
--    wild=450,
--    high=300,
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
    local mean = img.new(vgg_mean):view(3, 1, 1):expandAs(img)
    local perm = torch.LongTensor{3, 2, 1}
    return (img + mean):div(255):index(1, perm)
end
--modelG:zeroGradParameters()

for i = 1, opt.niter do
  real_uncropped,input= data:getBatch()
  --real=real_uncropped[{{},{},{1,1+93-1},{1,1+93-1}}]
  --real=real_uncropped[{{},{},{1,1+512-1},{1,1+512-1}}]
  --real=real_uncropped[{{},{},{1,opt.loadSize},{1,opt.loadSize}}]
  real=real_uncropped[{{},{},{1,opt.loadSize},{1,opt.loadSize}}]
  input = input:cuda()
  fake = modelG:forward(input)
  print("END of forward")
  for j=1,opt.batchSize do
    fake_p = deprocess(fake[j])
    fake_p[fake_p:gt(1)]=1
    fake_p[fake_p:lt(0)]=0

--    image.save(string.format('%s/raw_%04d.png',opt.result_path,cnt),image.toDisplayTensor(deprocess(real[j])))
    image.save(string.format('%s/fake_%04d.png',opt.result_path,cnt), fake_p)
    image.save(string.format('%s/raw_%04d.png',opt.result_path,cnt),deprocess(real[j]))

    cnt=cnt+1
  end
end




