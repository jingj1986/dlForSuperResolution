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
  size=9,
  loadSize=50,
  upsize=2,
--[[
  wild=512,
  high=384,
  t_folder='/home/user/data/video/youjidui/img/',
  --t_folder='/home/user/data/video/youjidui/from_phoenix/',
]]--

--[[
  wild=608,
  high=448,
  t_folder='/home/user/data/video/dileizhan/img/',
]]--

--[[
  wild=516,
  high=570,
--  t_folder='/home/user/data/video/Doctor/from_avs/png/',
  t_folder='/home/user/data/video/Doctor/neatvideo/',
]]--

--[[
  wild=608,
  high=448,
  t_folder='/home/user/data/video/dileizhan/img/',
]]--

--[[
  wild=720,
  high=480,
  t_folder='/home/user/data/video/VIDEO_TS/img/';
]]--

--[[
  wild=704,
  high=528,
  t_folder='/home/user/data/video/geju/img/',
]]--
  wild=352,
  high=288,
  t_folder='./test_pic/test/',
--  t_folder='/home/user/project/srgan_test/test_pic/test/',

  model_file='./model/test_model_t7',
  result_path='./result/'
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
print(dtype)
--local modelG = torch.load(opt.model_file):type(dtype)
--local modelG = torch.load(opt.model_file).model:type(dtype)
local modelG = util.load(opt.model_file, opt.gpu)
cutorch.setDevice(opt.gpu)
modelG = modelG:cuda()
print(modelG)
--local parametersG, gradientsG = modelG:getParameters()

cnt=1
function deprocess(img)
    return img:add(1):div(2)
end

for i =1, opt.size, 1 do
--for i =1, 1, 1 do
  local name = string.format("f-%05d.png", i)
  --local name = string.format("%03d00.png", i)
  --local name = string.format("foo-%06d.png", i)
  local path = opt.t_folder .. name
  local img = image.load(path, 3, 'float')
--  img = img:mul(255):add(-16):mul(255):div(219):div(255)

  limit_w = opt.wild/2 + 10

  test_input = torch.Tensor(1,1,opt.high,limit_w)
  test_input[1] = image.rgb2y(img)[{{},{},{1,limit_w}}]
  test_input = test_input:mul(2):add(-1)
  test_input = test_input:cuda()
  fake_tmp = modelG:forward(test_input)
  fake_1 = torch.Tensor(1,1,opt.high*opt.upsize,limit_w*opt.upsize)
  fake_1:copy(fake_tmp)

  test_input_2 = torch.Tensor(1,1,opt.high,limit_w)
  test_input_2[1] = image.rgb2y(img)[{{},{},{opt.wild- limit_w +1, opt.wild}}]
  test_input_2 = test_input_2:mul(2):add(-1)
  test_input_2 = test_input_2:cuda()
  fake_tmp = modelG:forward(test_input_2)
  fake_2 = torch.Tensor(1,1,opt.high*opt.upsize,limit_w*opt.upsize)
  fake_2:copy(fake_tmp)

--[[
  image.save(string.format('%s/%s',opt.result_path,"fake1.png"),image.toDisplayTensor(fake_1[1]))
  image.save(string.format('%s/%s',opt.result_path,"fake2.png"),image.toDisplayTensor(fake_2[1]))

  local fake = torch.Tensor(1, 1,opt.high*4, opt.wild*4)
  for h = 1, opt.high*4 do
    for w = 1, opt.wild*2 do
        fake[1][1][h][w] = fake_1[1][1][h][w]
    end
    for w = opt.wild*2, opt.wild*4 do
        idx = w - opt.wild*2 + 40
        fake[1][1][h][w] = fake_2[1][1][h][idx]
    end
  end
]]--
  local fake = torch.Tensor(1, 1,opt.high*opt.upsize, opt.wild*opt.upsize) 
  fake[{{},{},{},{1, limit_w*opt.upsize}}] = fake_1
  fake[{{},{},{},{opt.wild*(opt.upsize/2)-opt.upsize*10+1, opt.wild*opt.upsize}}] = fake_2

  for h = 1, opt.high*opt.upsize do
    for w = opt.wild*(opt.upsize/2)-opt.upsize*10, opt.wild*(opt.upsize/2) -(opt.upsize/2)*10 do
        fake[1][1][h][w] = fake_1[1][1][h][w]
    end
  end

  for j=1,opt.batchSize do 
    org = image.rgb2yuv(img)
    org = image.scale(org, opt.wild*opt.upsize, opt.high*opt.upsize, 'bilinear')

    fake_rgb = torch.Tensor(3, opt.high*opt.upsize,opt.wild*opt.upsize)
    fake_rgb[1]:copy(deprocess(fake[j][1]))
    fake_rgb[2]:copy(org[2])
    fake_rgb[3]:copy(org[3])
    fake_rgb = image.yuv2rgb(fake_rgb)
--[[
    fake_rgb = torch.Tensor(opt.high*opt.upsize,opt.wild*opt.upsize)
    fake_rgb:copy(deprocess(fake[j][1]))
    fake_rgb[fake_rgb:gt(1)]=1
    fake_rgb[fake_rgb:lt(0)]=0
]]--
    --image.save(string.format('%s/%s',opt.result_path,name),image.toDisplayTensor(fake_rgb))
    image.save(string.format('%s/%s',opt.result_path,name),fake_rgb)
--    image.save(string.format('%s/%s',opt.result_path,name),deprocess(fake[j][1]))
    --image.save(string.format('%s/%s',opt.result_path,name),fake[j][1])
    print(cnt .. ": " .. name)
    cnt=cnt+1
  end
end
