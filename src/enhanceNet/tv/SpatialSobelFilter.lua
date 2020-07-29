--- nn.SpatialSobelFilter
-- Input size: B, 1, H, W
-- Output size: B, 2, H, W

local SpatialSobelFilter, parent = torch.class('nn.SpatialSobelFilter', 'nn.Module')

local function makeSobelKernel()
    -- make x-, y- direction kernel, code borrowed from
    -- https://github.com/torch/demos/blob/e125c4854bea225bf1395bd834d6f305d35f226c/attention/attention.lua
    local kx= torch.Tensor(3,3)
    local a = kx:storage()
    a[1]=-1; a[2]=-2; a[3]=-1;
    a[4]=0;  a[5]=0;  a[6]=0;
    a[7]=1;  a[8]=2;  a[9]=1;

    local ky= torch.Tensor(3,3)
    local b = ky:storage()
    b[1]=-1; b[2]=0; b[3]=1;
    b[4]=-2; b[5]=0; b[6]=2;
    b[7]=-1; b[8]=0; b[9]=1;

    -- convnet to convolution module data layout
    local nInputPlane = 1 -- only one image
    local nOutputPlane = 2 -- x-, y- directional gradients
    local sk = torch.Tensor(nOutputPlane,nInputPlane,3,3)
    sk[1][nInputPlane]:copy(kx)
    sk[2][nInputPlane]:copy(ky)
    return sk
end

-- class def
function SpatialSobelFilter:__init()
    parent.__init(self)

    -- backend convolution module
    local ks = 3
    local stride = 1
    local pad = 1
    self.mconv = cudnn.SpatialConvolution(1,2, ks,ks, stride,stride, pad,pad)
    self.mconv:noBias() -- pure filter

    -- Sobel kernel
    self.mconv.weight:copy( makeSobelKernel() )

    -- must work with cuda tensor
    parent.cuda(self)
    self.mconv:cuda()
end

function SpatialSobelFilter:updateOutput(input)
    self.output = self.mconv:updateOutput(input)
    return self.output
end

function SpatialSobelFilter:updateGradInput(input, gradOutput)
    self.gradInput = self.mconv:updateGradInput(input, gradOutput)
    return self.gradInput
end

-- Okay with default accGradParameters()
