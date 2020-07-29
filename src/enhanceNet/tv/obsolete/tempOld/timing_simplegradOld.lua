require'cutorch'
require'tvnorm-nn'

B = 16
H = 400
W = 500

-- input
input = torch.Tensor(B, 1, H, W):normal(0,1):cuda()

-- gradOutput
gradOutput = torch.Tensor(B, 2, H, W):normal(0,1):cuda()

nloop = 3

function timing_module(input, gradOutput, m)
    cutorch.synchronize()
    local time

    -- fprop
    m:forward(input) -- warm up
    time = torch.tic()
    for i = 1, nloop do
        m:forward(input)
        cutorch.synchronize()
    end
    time = torch.toc(time)
    print(torch.type(m) .. ' fprop time ' .. time/nloop)

    -- bprop
    m:backward(input, gradOutput) -- warm up
    time = torch.tic()
    for i = 1, nloop do
        m:backward(input, gradOutput)
        cutorch.synchronize()
    end
    time = torch.toc(time)
    print(torch.type(m) .. ' bprop time ' .. time/nloop)

end

print('batch size = ' .. B)
m = nn.SpatialSimpleGradFilterOld()
timing_module(input, gradOutput, m)

output = m:forward(input)
gradInput = m:backward(input, gradOutput)