require 'torch'
require 'nn'

local crit, parent = torch.class('nn.AdversarialCriterion', 'nn.Criterion')

function crit:__init(args)
    parent.__init(self)
end

function crit:updateOutput(input, target)
    
end
