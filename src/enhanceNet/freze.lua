require 'nn'

function freze_N(net, startLayer, endLayer)

    function freze(net, count)
        for i = 1, #net.modules do
            local m = net.modules[i]
            if m.__typename == 'nn.Sequential' or m.__typename == 'nn.ConcatTable' then
                count = freze(m, count)
            end
            if m.__typename == 'nn.SpatialConvolution' or m.__typename == 'nn.SpatialBatchNormalization' then
                if count >= startLayer then
                    m.accGradParameters = function() end
                    m.updateParameters = function() end
                end
                count = count + 1
                if count >= endLayer then
                    return count
--                    break
                end
            end
--            if count >= endLayer then
--                return count
--            end
        end
        return count
    end
    freze(net, 0)
    return net
end

return freze_N
