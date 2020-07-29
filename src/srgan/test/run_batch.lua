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
  upsize=1,
  chan = 3,
  size=12,
  loadSize=50,

  --wild=352,
  wild=704,
  high=576,
--  wild = 480,
--  high = 320,
  t_folder='./test_pic/test/',
--  t_folder='/data/images_lib/video/from_hangtian/out/',
--  t_folder = '/home/user/data/video/xiaobingzhangga/avs/xbzg_deblock/',
  --t_folder='/home/user/data/video/xiaobingzhangga/avs/tmp/',

  model_file='./model/test_model_t7',
  result_path='./result/'
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)

local modelG = torch.load(opt.model_file):type(dtype)
print(modelG)
cutorch.setDevice(opt.gpu)
modelG = modelG:cuda()
--local parametersG, gradientsG = modelG:getParameters()

cnt=1
local vgg_mean = {103.939, 116.779, 123.68}
function preprocess(img)
--    local mean = img.new(vgg_mean):view(3, 1, 1):expandAs(img)
--    local perm = torch.LongTensor{3, 2, 1}
--    return img:index(1, perm):mul(255):add(-1, mean)
    return img:mul(2):add(-1)
end
function deprocess(img)
--    local mean = img.new(vgg_mean):view(3, 1, 1):expandAs(img)
--    local perm = torch.LongTensor{3, 2, 1}
--    return (img + mean):div(255):index(1, perm)
    return img:add(1):div(2)
end

--for i = 1, opt.niter do
--print(data:size())
for i = 1, opt.size, 1 do
  local name = string.format("f-%05d.png", i)
--  local name = string.format("016%d.png", i)
  local path = opt.t_folder .. name
  local img = image.load(path, 3, 'float')
  img = image.scale(img, opt.wild, opt.high, 'bicubic')

  --test_input = torch.Tensor(1,3,opt.high/opt.scale,opt.wild/opt.scale)
  test_input = torch.Tensor(1,3,opt.high,opt.wild)
  test_input[1] = preprocess(img)
  test_input = test_input:cuda()
  fake = modelG:forward(test_input)
  for j=1,opt.batchSize do 
    img_tmp = deprocess(fake[j])
    img_tmp[img_tmp:gt(1)] = 1
    img_tmp[img_tmp:lt(0)] = 0
    image.save(string.format('%s/%s',opt.result_path,name), img_tmp)
    print(cnt .. ": " .. name)
    cnt=cnt+1
  end
end
