require 'torch'
require 'nn'

require 'utils.ContentLoss'
local layer_utils = require 'utils.layer_utils'


local crit, parent = torch.class('nn.PerceptualCriterion', 'nn.Criterion')


--[[
Input: args is a table with the following keys:
- cnn: A network giving the base CNN.
- content_layers: An array of layer strings
- content_weights: A list of the same length as content_layers
- loss_type: Either "L2", or "SmoothL1"
--]]
function crit:__init(args)
  args.content_layers = args.content_layers or {}
  self.net = args.cnn
  self.net:evaluate()
  self.content_loss_layers = {}
  -- Set up content loss layers
  for i, layer_string in ipairs(args.content_layers) do
    local weight = args.content_weights[i]
    local content_loss_layer = nn.ContentLoss(weight, args.loss_type)
    layer_utils.insert_after(self.net, layer_string, content_loss_layer)
    table.insert(self.content_loss_layers, content_loss_layer)
  end

  layer_utils.trim_network(self.net)
  self.grad_net_output = torch.Tensor()

end


--[[
target: Tensor of shape (N, 3, H, W) giving pixels for content target images
--]]
function crit:setContentTarget(target)
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer:setMode('capture')
  end
  self.net:forward(target)
end

function crit:setContentWeight(weight)
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer.strength = weight
  end
end


--[[
Inputs:
- input: Tensor of shape (N, 3, H, W) giving pixels for generated images
- target: Tensor of shape (N, 3, H, W)
--]]
function crit:updateOutput(input, target)
  self:setContentTarget(target)

  -- Make sure to set all content and style loss layers to loss mode before
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer:setMode('loss')
  end

  local output = self.net:forward(input)
  -- Set up a tensor of zeros to pass as gradient to net in backward pass
  self.grad_net_output:resizeAs(output):zero()

  -- Go through and add up losses
  self.total_content_loss = 0
  self.content_losses = {}
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    self.total_content_loss = self.total_content_loss + content_loss_layer.loss
    table.insert(self.content_losses, content_loss_layer.loss)
  end
  self.output = self.total_content_loss
  return self.output
end


function crit:updateGradInput(input, target)
  self.gradInput = self.net:updateGradInput(input, self.grad_net_output)
  return self.gradInput
end

