require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'utils.TotalVariation'

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
                        :add(layer(16))
                        :add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
                        :add(nn.SpatialBatchNormalization(64))
                        :add(nn.Identity()))
                :add(nn.CAddTable(true))
    end

    model=nn.Sequential()
    model:add(nn.SpatialConvolution(3,64,9,9,1,1,4,4))
    model:add(nn.ReLU())

    model:add(multLayers())
    --model:add(layer(16))

    --model:add(nn.SpatialFullConvolution(64,256,3,3,1,1,1,1))
    --model:add(nn.SpatialConvolution(64,256,3,3,1,1,1,1))
    --model:add(nn.PixelShuffle(2))
    --model:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
    model:add(nn.SpatialUpSamplingBilinear(2))
    --model:add(nn.ReLU())
    --model:add(nn.SpatialFullConvolution(64,256,3,3,1,1,1,1))
    --model:add(nn.SpatialConvolution(64,256,3,3,1,1,1,1))
    --model:add(nn.PixelShuffle(2))
    --model:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
    model:add(nn.SpatialUpSamplingBilinear(2))  
    --model:add(nn.ReLU())
    --model:add(nn.SpatialFullConvolution(64,3,7,7,1,1,3,3))
    model:add(nn.SpatialFullConvolution(64,3,9,9,1,1,4,4))
    --model:add(nn.Tanh())
    --model:add(nn.MulConstant(150))
    model:add(nn.TotalVariation(1e-6))
    return model
end

return createModel
