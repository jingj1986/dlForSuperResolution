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
  size=1,
  loadSize=50,
  upsize=2,

  wild=640,
  high=360,
  t_folder='/home/user/project/srgan_test/test_pic/test/',
  --t_folder='/home/user/data/video/youjidui/from_phoenix/',

--[[
  wild=608,
  high=448,
  t_folder='/home/user/data/video/dileizhan/img/',
]]--
  model_file='./model/test_model_t7',
  result_path='./result/'
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)

local modelG = torch.load(opt.model_file):type(dtype)
--local modelG = torch.load(opt.model_file).model:type(dtype)
cutorch.setDevice(opt.gpu)
modelG = modelG:cuda()
local parametersG, gradientsG = modelG:getParameters()

cnt=1
function deprocess(img)
    return img:add(1):div(2)
end

for i = 1, opt.size, 1 do
  local name = string.format("f-%05d.png", i)
  --local name = string.format("f-%05d.tif", i)
  local path = opt.t_folder .. name
  local img = image.load(path, 3, 'float')
  --img = img:mul(255):add(-16):mul(255):div(219):div(255)

  limit_w = opt.wild/2 + 10
  --limit_w = opt.wild/8 + 10

  test_input = torch.Tensor(1,1,opt.high,limit_w)
  test_input[1] = image.rgb2y(img)[{{},{},{1,limit_w}}]
  --test_input[1] = image.rgb2y(img)[{{},{},{opt.wild - limit_w +1, opt.wild}}]
  test_input = test_input:mul(2):add(-1)
  test_input = test_input:cuda()
  fake = modelG:forward(test_input)

  for j=1,opt.batchSize do  
    org = image.rgb2yuv(img)
    org = image.scale(org, opt.wild*opt.upsize, opt.high*opt.upsize, 'bilinear')

--[[
    fake_rgb = torch.Tensor(3, opt.high*opt.upsize,limit_w*opt.upsize)
    fake_rgb[1]:copy(deprocess(fake[j][1]))
    fake_rgb[2]:copy(org[2][{{},{1,limit_w*opt.upsize}}])
    fake_rgb[3]:copy(org[3][{{},{1,limit_w*opt.upsize}}])
    fake_rgb = image.yuv2rgb(fake_rgb)
]]--
    
    fake_rgb = torch.Tensor( opt.high*opt.upsize,limit_w*opt.upsize)
    fake_rgb:copy(deprocess(fake[j][1]))
    fake_rgb[fake_rgb:gt(1)]=1
    fake_rgb[fake_rgb:lt(0)]=0

    image.save(string.format('%s/%s',opt.result_path,name),image.toDisplayTensor(fake_rgb))
    print(cnt .. ": " .. name)
    cnt=cnt+1
  end
end
