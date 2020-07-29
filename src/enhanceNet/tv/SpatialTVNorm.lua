--- nn.SpatialTVNorm
-- Input size: B, C, H, W
-- Output size: B

local SpatialTVNorm, parent = torch.class('nn.SpatialTVNorm', 'nn.Sequential')

local function checkInputSize(input)
    assert(input:dim() == 4, "This module works on 4D tensor")
    return input:size(1), input:size(2), input:size(3), input:size(4)
end

function SpatialTVNorm:__init()
    parent.__init(self)

    local B, C, H, W = 1, 1, 2, 2 -- reset them at run-time
    local Hd, Wd = 1, 1
    -- B, C, H, W
    self:add( nn.View(B*C, 1, H, W) ) -- 1
    -- BC, 1, H, W
    self:add( nn.SpatialSimpleGradFilter() )
    -- BC, 2, H', W'
    self:add( nn.Square() )
    -- BC, 2, H', W'
    self:add( nn.Sum(2) )
    -- BC, H', W'
    self:add( nn.Sqrt() )
    -- BC, H', W'
    self:add( nn.View(B, 1, C*Hd, Wd) ) -- 6
    -- B, 1, C*H', W'
    self:add( cudnn.SpatialAveragePooling(Wd,C*Hd, Wd,C*Hd, 0,0) ) -- 7
    -- B, 1, 1, 1
    self:add( nn.View(B) ) -- 8
    -- B
end

function SpatialTVNorm:updateOutput(input)
    local B, C, H, W = checkInputSize(input)
    self:_resetSize(B, C, H ,W)
    return parent.updateOutput(self, input)
end

function SpatialTVNorm:updateGradInput(input, gradOutput)
    local B, C, H, W = checkInputSize(input)
    self:_resetSize(B, C, H, W)
    return parent.updateGradInput(self, input, gradOutput)
end

function SpatialTVNorm:_resetSize(B, C, H, W)
    local Hd, Wd = H-1, W-1

    -- View
    self.modules[1]:resetSize(B*C, 1, H, W)

    -- View
    self.modules[6]:resetSize(B, 1, C*Hd, Wd)

    -- SpatialAveragePooling
    self.modules[7].kW = Wd
    self.modules[7].kH = C*Hd
    self.modules[7].dW = Wd
    self.modules[7].dH = C*Hd

    -- View
    self.modules[8]:resetSize(B)
end