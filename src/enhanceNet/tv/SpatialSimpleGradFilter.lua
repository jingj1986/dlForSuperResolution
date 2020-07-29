--- nn.SpatialSimpleGradFilter
-- Input size: B, 1, H, W
-- Output size: B, 2, H-1, W-1
require 'cudnn'

local SpatialSimpleGradFilter, parent = torch.class('nn.SpatialSimpleGradFilter', 'nn.Module')

local function makeSimpleGradKernel()
    -- make x-, y- direction kernel, code borrowed from
    local kx= torch.Tensor(2,2)
    local a = kx:storage()
    a[1] = -1; a[2] = 1
    a[3] = 0;  a[4] = 0;

    local ky= torch.Tensor(2,2)
    local b = ky:storage()
    b[1] = -1; b[2] = 0;
    b[3] = 1;  b[4] = 0;

    -- convnet to convolution module data layout
    local nInputPlane = 1 -- only one image
    local nOutputPlane = 2 -- x-, y- directional gradients
    local sk = torch.Tensor(nOutputPlane,nInputPlane,2,2)
    sk[1][nInputPlane]:copy(kx)
    sk[2][nInputPlane]:copy(ky)
    return sk
end

-- class def
function SpatialSimpleGradFilter:__init()
    parent.__init(self)

    -- backend convolution module
    local ks = 2
    local stride = 1
    local pad = 0
    self.mconv = cudnn.SpatialConvolution(1,2, ks,ks, stride,stride, pad,pad)
    self.mconv:noBias() -- pure filter

    -- Sobel kernel
    self.mconv.weight:copy( makeSimpleGradKernel() )

    -- must work with cuda tensor
    parent.cuda(self)
    self.mconv:cuda()
end

function SpatialSimpleGradFilter:updateOutput(input)
    self.output = self.mconv:updateOutput(input)
    return self.output
end

function SpatialSimpleGradFilter:updateGradInput(input, gradOutput)
    self.gradInput = self.mconv:updateGradInput(input, gradOutput)
    return self.gradInput
end

-- Okay with default accGradParameters()
