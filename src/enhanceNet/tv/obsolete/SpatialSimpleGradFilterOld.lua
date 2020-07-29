--- nn.SpatialSimpleGradFilterOld
-- Input size: B, 1, H, W
-- Output size: B, 2, H, W

local SpatialSimpleGradFilterOld, parent = torch.class('nn.SpatialSimpleGradFilterOld', 'nn.Module')

local function makeSimpleGradKernel()
    -- make x-, y- direction kernel, code borrowed from
    local kx= torch.Tensor(3,3)
    local a = kx:storage()
    a[1]=0;  a[2]=0   a[3]=0;
    a[4]=0;  a[5]=-1; a[6]=1;
    a[7]=0;  a[8]=0;  a[9]=0;

    local ky= torch.Tensor(3,3)
    local b = ky:storage()
    b[1]=0; b[2]=0; b[3]=0;
    b[4]=0; b[5]=-1; b[6]=0;
    b[7]=0; b[8]=1; b[9]=0;

    -- convnet to convolution module data layout
    local nInputPlane = 1 -- only one image
    local nOutputPlane = 2 -- x-, y- directional gradients
    local sk = torch.Tensor(nOutputPlane,nInputPlane,3,3)
    sk[1][nInputPlane]:copy(kx)
    sk[2][nInputPlane]:copy(ky)
    return sk
end

-- class def
function SpatialSimpleGradFilterOld:__init()
    parent.__init(self)

    -- backend convolution module
    local ks = 3
    local stride = 1
    local pad = 1
    self.mconv = cudnn.SpatialConvolution(1,2, ks,ks, stride,stride, pad,pad)
    self.mconv:noBias() -- pure filter

    -- Sobel kernel
    self.mconv.weight:copy( makeSimpleGradKernel() )

    -- must work with cuda tensor
    parent.cuda(self)
    self.mconv:cuda()
end

function SpatialSimpleGradFilterOld:updateOutput(input)
    self.output = self.mconv:updateOutput(input)
    return self.output
end

function SpatialSimpleGradFilterOld:updateGradInput(input, gradOutput)
    self.gradInput = self.mconv:updateGradInput(input, gradOutput)
    return self.gradInput
end

-- Okay with default accGradParameters()
