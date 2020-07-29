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
local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.loadSize}

local function loadImage(path)
   --local input = image.load(path, 3, 'float')
   local isok,input = pcall(image.load, path, 3, 'float')
   if isok == false or input:dim() ~= 3 or input == nil then
    print("err image" .. path)
    return nil
   end
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   local iW = input:size(3)
   local iH = input:size(2)
  

   if iW < iH then
      input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
   else
      input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
   end

   return input
end


local vgg_mean = {103.939, 116.779, 123.68}
local function check_input(img)
    assert(img:dim() == 3, 'img must be C x H x W')
    assert(img:size(1) == 3, 'img must have three channels')
end

function preprocess(img)
    local mean = img.new(vgg_mean):view(3, 1, 1):expandAs(img)
    local perm = torch.LongTensor{3, 2, 1}
    return img:index(1, perm):mul(255):add(-1, mean) 
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded
-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   collectgarbage()
   local input = loadImage(path)
   if not input then
        return nil, nil
   end
   local input_1 =  image.scale(input,opt.wild,opt.high,'bicubic')
   
   --local input_y=image.rgb2y(input)
   --local input_y=image.rgb2y(input_tmp)
   
   local input_2=image.scale(input,opt.wild/opt.scale,opt.high/opt.scale,'bicubic')

   --local input_2=image.scale(input_2,opt.loadSize,opt.loadSize,'bicubic')
   --local input_2=image.scale(input,opt.loadSize/opt.scale,opt.loadSize/opt.scale,'bicubic')
   --local input_2=image.scale(input_2,opt.loadSize,opt.loadSize,'bicubic')
   --local input_y2=image.rgb2y(input) 
    
   return preprocess(input_1), preprocess(input_2)
--    return input_1, input_2
end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {3, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {3, sampleSize[2], sampleSize[2]}
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {opt.data},
      loadSize = {3, loadSize[2], loadSize[2]},
      sampleSize = {3, sampleSize[2], sampleSize[2]},
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
