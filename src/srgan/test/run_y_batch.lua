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
  scale=4,
  chan = 3,
  size=976,
  loadSize=50,
  --wild=640,
  --high=256,
  --wild=352,
  --wild=384,
  --high=288,
  --wild=512,
  --high=384,
  --t_folder='/home/user/data/video/qianlong/img/',
  --t_folder='/home/user/data/video/xiaobingzhangga/img/',
  --t_folder='/home/user/sample/test_res/img/',
  --t_folder='/data/tmp/xiju/pic/',
  --t_folder='/home/user/data/video/youjidui/img/',
--[[
  wild = 608,
  high = 448,
  t_folder='/home/user/data/video/dileizhan/img/',
]]--
  wild = 352,
  high = 288,
--  t_folder='/home/user/data/test/xbzg_1/',
--  t_folder='./test_pic/test/',
  t_folder='/home/user/data/video/xiaobingzhangga/png/',

--[[
  wild=516,
  high=570,
  t_folder='/home/user/data/video/Doctor/from_avs/png/',
]]--
--[[
  wild=352,
  high=288,
  t_folder='/home/user/data/video/xiaobingzhangga/avs/png/',
]]--

--[[
  wild=352,
  high=288,
  --t_folder='/home/user/data/video/xiaobingzhangga/png/',
  --t_folder='/home/user/data/video/xiaobingzhangga/avs/png/',
  --t_folder='/home/user/data/video/xiju/f8/png/',
  t_folder='/home/user/data/video/baimaonv/png/',
]]--

--[[
  wild=352,
  high=264,
  t_folder = '/home/user/data/video/cimuqu/png/',
]]--

  model_file='./model/test_model_t7',
  result_path='./result/'
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

--local DataLoader = paths.dofile('data/data.lua')
--data = DataLoader.new(opt.nThreads,  opt)

local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)

print(dtype)
--local modelG = util.load(opt.model_file, opt.gpu)
local modelG = torch.load(opt.model_file):type(dtype)
print(modelG)
--local modelG = torch.load(opt.model_file).model:type(dtype)
cutorch.setDevice(opt.gpu)
modelG = modelG:cuda()
local parametersG, gradientsG = modelG:getParameters()

cnt=1
function deprocess(img)
    return img:add(1):div(2)
end

--for i = 1, opt.niter do
--print(data:size())
for i = 1, opt.size, 1 do
--for i = 2000, opt.size+2000, 1 do
  local name = string.format("f-%05d.png", i)
--  local name = string.format("%05d.png", i)
  --local name = string.format("foo-%06d.png", i)
  local path = opt.t_folder .. name
  local img = image.load(path, 3, 'float')

--  real_uncropped,input,name= data:getBatch()
--  real=real_uncropped[{{},{},{1,opt.high},{1,opt.wild}}]

  test_input = torch.Tensor(1,1,opt.high,opt.wild)
  test_input[1] = image.rgb2y(img)
  test_input = test_input:mul(2):add(-1)
    
  test_input = test_input:cuda()
  fake = modelG:forward(test_input)
--  print("END of forward")
  for j=1,opt.batchSize do  
--    org = deprocess(input[1])

    org = image.rgb2yuv(img)
    org = image.scale(org, opt.wild*opt.scale, opt.high*opt.scale, 'bilinear')
    fake_rgb = torch.Tensor(3, opt.high*opt.scale,opt.wild*opt.scale)
    --org = image.scale(org, opt.wild*4, opt.high*4, 'bilinear')
    --fake_rgb = torch.Tensor(3, opt.high*4,opt.wild*4)
    fake_rgb[1]:copy(deprocess(fake[j][1]))
    fake_rgb[2]:copy(org[2])
    fake_rgb[3]:copy(org[3])
    fake_rgb = image.yuv2rgb(fake_rgb)
    fake_rgb[fake_rgb:gt(1)]=1
    fake_rgb[fake_rgb:lt(0)]=0

--    image.save(string.format('%s/raw_%04d.png',opt.result_path,cnt),image.toDisplayTensor(deprocess(real[j])))
--    image.save(string.format('%s/fake_%05d.png',opt.result_path,cnt),image.toDisplayTensor(fake_rgb))
    --image.save(string.format('%s/%s',opt.result_path,name),image.toDisplayTensor(fake_rgb))
    image.save(string.format('%s/%s',opt.result_path,name),fake_rgb)
--    image.save(string.format('%s/%05d.png',opt.result_path,i),fake_rgb)
--    image.save(string.format('%s/%s',opt.result_path,name),image.toDisplayTensor(fake))
    --image.save(string.format('%s/%s',opt.result_path,name),deprocess(fake[j][1]))
    print(cnt .. ": " .. name)
    cnt=cnt+1
  end
end
