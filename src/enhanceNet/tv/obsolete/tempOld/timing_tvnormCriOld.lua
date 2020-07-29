require'cutorch'
require'tvnorm-nn'

B = 16
C = 2
H = 400
W = 500

-- input
input = torch.Tensor(B, C, H, W):normal(0,1):cuda()
target = nil

nloop = 3

function timing_module(input, target, m)
    cutorch.synchronize()
    local time

    -- fprop
    m:forward(input, target) -- warm up
    time = torch.tic()
    for i = 1, nloop do
        m:forward(input, target)
        cutorch.synchronize()
    end
    time = torch.toc(time)
    print(torch.type(m) .. ' fprop time ' .. time/nloop)

    -- bprop
    m:backward(input, target) -- warm up
    time = torch.tic()
    for i = 1, nloop do
        m:backward(input, target)
        cutorch.synchronize()
    end
    time = torch.toc(time)
    print(torch.type(m) .. ' bprop time ' .. time/nloop)

end

print('batch size = ' .. B)
m = nn.SpatialTVNormCriterionOld('sobel'):cuda()
timing_module(input, target, m)

output = m:forward(input, target)
gradInput = m:backward(input, target)

m2= nn.SpatialTVNormCriterionOld('simple'):cuda()
timing_module(input, target, m2)

output2 = m2:forward(input, target)
gradInput2 = m2:backward(input, target)