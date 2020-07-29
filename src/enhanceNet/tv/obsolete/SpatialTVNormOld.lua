--- nn.SpatialTVNormOld
-- Input size: B, C, H, W
-- Output size: B

local SpatialTVNormOld, parent = torch.class('nn.SpatialTVNormOld', 'nn.Sequential')

local function checkInputSize(input)
    assert(input:dim() == 4, "This module works on 4D tensor")
    return input:size(1), input:size(2), input:size(3), input:size(4)
end

function SpatialTVNormOld:__init(kerType)
    parent.__init(self)

    local mFilter
    kerType = kerType or 'sobel'
    if kerType == 'sobel' then
        mFilter = nn.SpatialSobelFilter()
    elseif kerType == 'simple' then
        mFilter = nn.SpatialSimpleGradFilterOld()
    else
        error('unknown kerType '..kerType)
    end

    local B, C, H, W = 1, 1, 1, 1 -- reset them at run-time
    -- B, C, H, W
    self:add( nn.View(B*C, 1, H, W) ) -- 1
    -- BC, 1, H, W
    self:add( mFilter )
    -- BC, 2, H, W
    self:add( nn.Abs() )
    -- BC, 2, H, W
    self:add( nn.View(B, 1, C*2*H, W) ) -- 4
    -- B, 1, C*2*H, W
    self:add( cudnn.SpatialAveragePooling(W,C*2*H, W,C*2*H, 0,0) ) -- 5
    -- B, 1, 1, 1
    self:add( nn.View(B) ) -- 6
    -- B
end

function SpatialTVNormOld:updateOutput(input)
    local B, C, H, W = checkInputSize(input)
    self:_resetSize(B, C, H ,W)
    return parent.updateOutput(self, input)
end

function SpatialTVNormOld:updateGradInput(input, gradOutput)
    local B, C, H, W = checkInputSize(input)
    self:_resetSize(B, C, H, W)
    return parent.updateGradInput(self, input, gradOutput)
end

function SpatialTVNormOld:_resetSize(B, C, H, W)
    -- do it in a dirty but fast way

    -- View
    self.modules[1]:resetSize(B*C, 1, H, W)

    -- View
    self.modules[4]:resetSize(B, 1, C*2*H, W)

    -- SpatialAveragePooling
    self.modules[5].kW = W
    self.modules[5].kH = C*2*H
    self.modules[5].dW = W
    self.modules[5].dH = C*2*H

    -- View
    self.modules[6]:resetSize(B)
end

function SpatialTVNormOld:_resetSizeDirtyButFast(B, C, H, W)
    error('shoul not call this function. bugs exist.')
    -- do it in a dirty but fast way

    -- View
    self.modules[1].size[1] = B*C
    self.modules[1].size[3] = H
    self.modules[1].size[4] = W

    -- View
    self.modules[4].size[1] = B
    self.modules[4].size[3] = C*2*H
    self.modules[4].size[4] = W

    -- SpatialAveragePooling
    self.modules[5].kW = W
    self.modules[5].kH = C*2*H
    self.modules[5].dW = W
    self.modules[5].dH = C*2*H

    -- View
    self.modules[6].size[1] = B
end