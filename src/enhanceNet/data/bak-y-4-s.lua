 --[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT') or opt.t_folder
if not paths.dirp(opt.data) then
    print('Did not find directory: '..opt.data)
    os.exit()
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local loadSize   = {opt.chan, opt.loadSize}
local sampleSize = {opt.chan, opt.loadSize}

local function loadImage(path)
   --local input = image.load(path, 3, 'float')
   path_low =string.gsub(path, "high", "low")
   local isok,high = pcall(image.load, path, 3, 'float')
   local isok1,low = pcall(image.load, path_low, 3, 'float')

   if isok == false or isok1 == false or high:dim() ~= 3 or high == nil or low:dim() ~= 3 then
    print("err image" .. path)
    return nil
   end
   return high, low
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded
local vgg_mean = {103.939, 116.779, 123.68}

--[[
Preprocess an image before passing to a VGG model. We need to rescale from
[0, 1] to [0, 255], convert from RGB to BGR, and subtract the mean.

Input:
- img: Tensor of shape (N, C, H, W) giving a batch of images. Images 
]]
function preprocess(img)
  --check_input(img)
  local mean = img.new(vgg_mean):view(3, 1, 1):expandAs(img)
  local perm = torch.LongTensor{3, 2, 1}
  return img:index(1, perm):mul(255):add(-1, mean)
end


-- Undo VGG preprocessing
function deprocess(img)
  check_input(img)
  local mean = img.new(vgg_mean):view(1, 3, 1, 1):expandAs(img)
  local perm = torch.LongTensor{3, 2, 1}
  return (img + mean):div(255):index(2, perm)
end

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   collectgarbage()
   local high, low = loadImage(path)
   if not high then
        return nil, nil
   end

   local h = low:size(2)
   local w = low:size(3)
   local Xmin = math.floor(torch.uniform(0, w - opt.loadSize/4))
   local Ymin = math.floor(torch.uniform(0, h - opt.loadSize/4))

   local Xmax = Xmin + opt.loadSize/4
   local Ymax = Ymin + opt.loadSize/4

   local input_LR = image.crop(low, Xmin, Ymin, Xmax , Ymax)  
   local input_HR = image.crop(high,Xmin*4,Ymin*4,Xmax*4,Ymax*4)

--   local input_HR = preprocess(input_HR) 
--   local input_LR = preprocess(input_LR)
    local input_HR = image.rgb2y(input_HR)
    local input_LR = image.rgb2y(input_LR)
    input_HR = input_HR:mul(2):add(-1)
    img1 = input_LR:mul(255):mul(219):div(255):add(16)


    input_LR = input_LR:mul(2):add(-1)

   return input_HR,input_LR
    
end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {opt.chan, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {opt.chan, sampleSize[2], sampleSize[2]}
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {opt.data},
      loadSize = {opt.chan, loadSize[2], loadSize[2]},
      sampleSize = {opt.chan, sampleSize[2], sampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   print('saved metadata cache at', trainCache)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end
