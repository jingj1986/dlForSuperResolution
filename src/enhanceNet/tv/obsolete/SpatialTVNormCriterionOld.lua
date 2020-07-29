--- nn.SpatialTVNormCriterionOld, useful when nn.SpatialTVNorm is a regularizer
-- Input size: B, C, H, W
-- Target size: dummy
-- Output size: 1

local SpatialTVNormCriterionOld, parent = torch.class('nn.SpatialTVNormCriterionOld', 'nn.Criterion')

function SpatialTVNormCriterionOld:__init(kerType)
    parent.__init(self)

    self.mTVNorm = nn.SpatialTVNormOld(kerType)
    self.gradOutputConst = torch.Tensor(1):fill(1)
end

function SpatialTVNormCriterionOld:updateOutput(input, target)
    -- B, C, H, W
    local output = self.mTVNorm:updateOutput(input)
    -- B
    self.output = output:sum()/output:nElement() -- a lua number, averaged by batch size!
    return self.output
end

function SpatialTVNormCriterionOld:updateGradInput(input, target)
    -- reuse constant tensor if batch size unchanges
    local B = input:size(1)
    if self.gradOutputConst:type() ~= input:type() or self.gradOutputConst:size(1) ~= B then
        self.gradOutputConst:resize(B):fill(1/B) -- averaged by batch size!
    end

    self.gradInput = self.mTVNorm:updateGradInput(input, self.gradOutputConst)
    return self.gradInput
end