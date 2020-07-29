require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
--require 'utils.TotalVariation'

local function createModel()
      local function bottleneck()
          local convs=nn.Sequential()
          convs:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
          convs:add(nn.SpatialBatchNormalization(64))
          convs:add(nn.ReLU(true))
          convs:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
          convs:add(nn.SpatialBatchNormalization(64))
          local shortcut=nn.Identity()
          return nn.Sequential():add(nn.ConcatTable():add(convs):add(shortcut)):add(nn.CAddTable(true))
      end

    local function layer(count)
      local s=nn.Sequential()
      for i=1,count do
        s:add(bottleneck())
      end
      return s
    end
    
    local function multLayers()
        local s=nn.Sequential()
        return nn.Sequential()
                :add(nn.ConcatTable()
                        :add(layer(13))
                        :add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
                        :add(nn.SpatialBatchNormalization(64))
                        :add(nn.Identity()))
                :add(nn.CAddTable(true))
    end

    model=nn.Sequential()
    model:add(nn.SpatialConvolution(1,64,3,3,1,1,1,1))
    model:add(nn.ReLU())
    --model:add(layer(1))
    model:add(multLayers())
    --model:add(nn.SpatialFullConvolution(64,256,3,3,1,1,1,1))
    --model:add(nn.SpatialConvolution(64,256,3,3,1,1,1,1))
    --model:add(nn.PixelShuffle(2))
--[[
    model:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
    model:add(nn.SpatialUpSamplingBilinear(2))
    --model:add(nn.SpatialUpsamplingNearest(2))
    model:add(nn.ReLU())
    --model:add(nn.SpatialFullConvolution(64,256,3,3,1,1,1,1))
    --model:add(nn.SpatialConvolution(64,256,3,3,1,1,1,1))
    --model:add(nn.PixelShuffle(2))
    model:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
    model:add(nn.SpatialUpSamplingBilinear(2))  
    model:add(nn.ReLU())
]]--
    model:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.ReLU())
    model:add(nn.SpatialFullConvolution(64,1,3,3,1,1,1,1))
    --model:add(nn.TotalVariation(1e-6))
    return model
end

return createModel
