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
  scale=2,
  upsize=2,
  chan = 3,
  size=2,
  loadSize=50,

  wild=352,
  high=288,
--  t_folder='/home/user/data/video/xiaobingzhangga/avs/png/',
  t_folder='./test_pic/test/',
--  t_folder='/home/user/data/video/xiaobingzhangga/png/',
--  t_folder = '/home/user/data/video/xiaobingzhangga/avs/xbzg_deblock/',
  model_file='./model/test_model_t7',
--  model_file='./model/test_model_G',
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
    local mean = img.new(vgg_mean):view(3, 1, 1):expandAs(img)
    local perm = torch.LongTensor{3, 2, 1}
    return img:index(1, perm):mul(255):add(-1, mean)
end
function deprocess(img)
    local mean = img.new(vgg_mean):view(3, 1, 1):expandAs(img)
    local perm = torch.LongTensor{3, 2, 1}
    return (img + mean):div(255):index(1, perm)
end

--for i = 1, opt.niter do
--print(data:size())
--for i = 7000, opt.size+7000, 1 do
for i = 1, 9 do
  local name = string.format("f-%05d.png", i)
  local path = opt.t_folder .. name
  local img = image.load(path, 3, 'float')

  limit_w = opt.wild/2 + 10

  test_input = torch.Tensor(1,3,opt.high,limit_w)
  img = preprocess(img)
--  test_input[1] = preprocess(img)[{{},{},{1,limit_w}}]
  test_input[1] = img[{{},{},{1,limit_w}}]
  test_input = test_input:cuda()
  fake_tmp = modelG:forward(test_input)
  fake_1 = torch.Tensor(1,3,opt.high*opt.upsize,limit_w*opt.upsize)
  fake_1:copy(fake_tmp)

  test_input_2 = torch.Tensor(1,3,opt.high,limit_w)
--  test_input_2[1] = preprocess(img)[{{},{},{opt.wild- limit_w +1, opt.wild}}]
  test_input_2[1] = img[{{},{},{opt.wild- limit_w +1, opt.wild}}]
  test_input_2 = test_input_2:cuda()
  fake_tmp = modelG:forward(test_input_2)
  fake_2 = torch.Tensor(1,3,opt.high*opt.upsize,limit_w*opt.upsize)
  fake_2:copy(fake_tmp)

  local fake = torch.Tensor(1, 3,opt.high*opt.upsize, opt.wild*opt.upsize) 
  fake[{{},{},{},{1, limit_w*opt.upsize}}] = fake_1
  fake[{{},{},{},{opt.wild*(opt.upsize/2)-opt.upsize*10+1, opt.wild*opt.upsize}}] = fake_2

  for h = 1, opt.high*opt.upsize do
    for w = opt.wild*(opt.upsize/2)-opt.upsize*10, opt.wild*(opt.upsize/2) -(opt.upsize/2)*10 do
        fake[1][1][h][w] = fake_1[1][1][h][w]
        fake[1][2][h][w] = fake_1[1][2][h][w]
        fake[1][3][h][w] = fake_1[1][3][h][w]
    end
  end

  for j=1,opt.batchSize do
    --image.save(string.format('%s/%s',opt.result_path,name),image.rgb2y(deprocess(fake[j])))
    image.save(string.format('%s/%s',opt.result_path,name),deprocess(fake[j]))
    print(cnt .. ": " .. name)
    cnt = cnt+1 
  end



--[[
  test_input = torch.Tensor(1,3,opt.high,opt.wild)
  test_input[1] = preprocess(img)
  test_input = test_input:cuda()
  fake = modelG:forward(test_input)
  for j=1,opt.batchSize do  
    image.save(string.format('%s/%s',opt.result_path,name),deprocess(fake[j][1]))
    print(cnt .. ": " .. name)
    cnt=cnt+1
  end
]]--
end
