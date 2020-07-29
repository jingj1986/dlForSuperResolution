require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'utils.TotalVariation'

local function createModel()
      local function bottleneck()
          local convs=nn.Sequential()
          convs:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))
--          convs:add(nn.SpatialBatchNormalization(64))
          convs:add(nn.ReLU(true))
          convs:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))
--          convs:add(nn.SpatialBatchNormalization(64))
          convs:add(nn.MulConstant(0.1))
          local shortcut=nn.Identity()
          return nn.Sequential():add(nn.ConcatTable():add(convs):add(shortcut)):add(nn.CAddTable(true))
      end

    local function layer(count)
      local s=nn.Sequential()
      for i=1,count do
        s:add(bottleneck())
      end
      s:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))
      return s
    end

    local function upsample()
        local s = nn.Sequential()
        --model:add(nn.SpatialFullConvolution(256,256,3,3,1,1,1,1))
        --model:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
        --model:add(nn.SpatialUpSamplingBilinear(2))
        --model:add(nn.SpatialUpSamplingNearest(2))
        --model:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
        --model:add(nn.ReLU())

        s:add(nn.SpatialConvolution(256,1024,3,3,1,1,1,1))
        s:add(nn.PixelShuffle(2))

        --model:add(nn.SpatialFullConvolution(64,256,3,3,1,1,1,1))
        --model:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))
        --model:add(nn.PixelShuffle(2))
        --model:add(nn.SpatialConvolution(64,256,3,3,1,1,1,1))
        --model:add(nn.PixelShuffle(2))
        --model:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
        --model:add(nn.SpatialUpSamplingBilinear(2)) 
        --model:add(nn.SpatialUpSamplingNearest(2))
        --model:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
        --model:add(nn.ReLU())
       return s
    end

    local function multLayers()
        local s=nn.Sequential()
        return nn.Sequential()
                :add(nn.ConcatTable()
                        :add(layer(36))
                        --:add(nn.SpatialBatchNormalization(64))
                        :add(nn.Identity()))
                :add(nn.CAddTable(true))
    end

    model=nn.Sequential()
    model:add(nn.SpatialConvolution(3,256,3,3,1,1,1,1))
    --model:add(nn.ReLU())

    model:add(multLayers())
    model:add(upsample())

    model:add(nn.SpatialConvolution(256,3,3,3,1,1,1,1))
    return model
end

return createModel
