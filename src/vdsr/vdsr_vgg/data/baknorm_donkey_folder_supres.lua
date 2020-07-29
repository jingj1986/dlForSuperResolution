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
   local isok,input = pcall(image.load, path, 3, 'float')
   if isok == false or input:dim() ~= 3 or input == nil then
    print("err image" .. path)
    return nil
   end
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   local iW = input:size(3)
   local iH = input:size(2)

   if iW < opt.loadSize or iH < opt.loadSize then
     print("too small image " .. path )
     return nil
   end
   
   --[[
   if iW < iH then
      input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
   else
      input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
   end
   --]]

   return input
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
   local h = input:size(2)
   local w = input:size(3)
   local Xmin = math.floor(torch.uniform(0, w - opt.loadSize))
   local Ymin = math.floor(torch.uniform(0, h - opt.loadSize))

   local Xmax = Xmin + opt.loadSize
   local Ymax = Ymin + opt.loadSize
   --local input_1 =  image.scale(input,opt.loadSize,opt.loadSize,'bicubic')
   --local input_y=image.rgb2y(input)
   --local input_y=image.rgb2y(input_tmp)
   local input_HR = image.crop(input, Xmin, Ymin, Xmax, Ymax) 
   --local input_HR = image.scale(input,opt.loadSize,opt.loadSize,'bicubic')
   local input_LR = image.scale(input_HR,opt.loadSize/2,opt.loadSize/2,'bicubic')
   local input_LR = image.scale(input_LR,opt.loadSize,opt.loadSize,'bicubic')
   --local input_y2=image.rgb2y(input)
  
   --return image.rgb2y(input_HR),image.rgb2y(input_LR)
    return image.rgb2y(input_HR):mul(2):add(-1),image.rgb2y(input_LR):mul(2):add(-1)
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
