require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'

require 'utils.TotalVariation'
require 'utils.InstanceNormalization'

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

    model = nn.Sequential()

    main=nn.Sequential()
    main:add(nn.SpatialConvolution(1,64,3,3,1,1,1,1))
--    main:add(nn.SpatialConvolution(3,64,9,9,1,1,4,4))
    main:add(nn.SpatialBatchNormalization(64))
    main:add(nn.ReLU())

    main:add(layer(7))
--[[
    main:add(nn.SpatialUpSamplingNearest(2))
    main:add(nn.SpatialBatchNormalization(64))
    main:add(nn.ReLU())

    main:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
    main:add(nn.SpatialBatchNormalization(64))
    main:add(nn.ReLU())

]]--
    main:add(nn.SpatialUpSamplingNearest(2))
    main:add(nn.SpatialBatchNormalization(64))
    main:add(nn.ReLU())
    main:add(nn.SpatialUpSamplingNearest(2))
    main:add(nn.SpatialBatchNormalization(64))
    main:add(nn.ReLU())


--    main:add(nn.SpatialConvolution(64,3,9,9,1,1,4,4))
    main:add(nn.SpatialConvolution(64,1,3,3,1,1,1,1))
--[[
    main:add(nn.SpatialBatchNormalization(3))
    main:add(nn.ReLU())
    main:add(nn.SpatialConvolution(3,3,3,3,1,1,1,1))
    
    main:add(nn.Tanh())
    main:add(nn.MulConstant(150))
    main:add(nn.TotalVariation(1e-6))
]]--

    local shortcut = nn.Sequential()
    shortcut:add(nn.SpatialUpSamplingBilinear(4))

    local concat = nn.ConcatTable()
    --concat:add(main):add(nn.Identity())
    --concat:add(main):add(nn.Identity()):add(nn.TotalVariation(1e-1))
    concat:add(main):add(shortcut)
    model:add(concat)
    model:add(nn.CAddTable(true))
--[[
    concat:add(main):add(res)
        nn.Sequential():add(nn.SpatialUpSamplingBilinear(4)))
    concat:add(nn.Concat(2)
              :add(main)
              :add(nn.SpatialUpSamplingBilinear(4)))
]]--
    --model:add(concat)
--    return nn.Sequential():add(model)
    return model
--    return main

end

return createModel
