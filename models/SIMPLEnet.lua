--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--  Architecture borrowed from FlowNet:Simple
--
--  Fischer, Philipp, et al. "Flownet: Learning optical flow with convolutional networks."
--  arXiv preprint arXiv:1504.06852 (2015).
--

--require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'math'
require 'stn'

local nninit = require 'nninit'

local numChannels       = 3

local Conv      = cudnn.SpatialConvolution
local deConv    = cudnn.SpatialFullConvolution
local ReLU      = cudnn.ReLU
local MaxPool   = cudnn.SpatialMaxPooling

local function createModel(opt)

    local model     = nn.Sequential()


    model:add(Conv(2, 16, 5, 5))
    model:add(MaxPool(2,2,2,2))
    model:add(ReLU())
    model:add(Conv(16, 16, 5, 5))
    model:add(MaxPool(2,2,2,2))
    model:add(ReLU())
    model:add(Conv(16, 32, 5, 5))
    model:add(MaxPool(2,2,2,2))
    model:add(ReLU())
    model:add(Conv(32, 32, 5, 5))
    model:add(MaxPool(2,2,2,2))
    model:add(ReLU())
    model:add(Conv(32, 64, 3, 3, 1, 1, 1, 1))
    model:add(MaxPool(2,2,2,2))
    model:add(ReLU())
    model:add(Conv(64, 64, 3, 3, 1, 1, 1, 1))
    model:add(MaxPool(2,2,2,2))
    model:add(ReLU())


    model:add(nn.View(64*3*3))
    model:add(nn.Linear(64*3*3,64))
    model:add(ReLU())

    -- we initialize the output layer so it gives the identity transform
    local outLayer = nn.Linear(64,6)
    -- local outLayer = nn.Linear(20,6)
    outLayer.weight:fill(0)
    local bias = torch.FloatTensor(6):fill(0)
    bias[1]=1
    bias[5]=1
    outLayer.bias:copy(bias)
    model:add(outLayer)

    --  there we generate the grids
    model:add(nn.View(2,3))

    -- local model_2   = nn.Sequential()    
    -- local parnet = nn.ParallelTable()
    -- local trans = nn.Sequential()
    -- trans:add(nn.Identity())
    -- trans:add(nn.Transpose({2,3},{3,4}))
    -- parnet:add(trans)
    -- parnet:add(nn.AffineGridGeneratorBHWD(256,256))
    -- model_2:add(parnet)
    -- model_2:add(nn.BilinearSamplerBHWD())
    -- model_2:add(nn.Transpose({3,4},{2,3}))


   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end

--       local function ConvInit(name)
--       for k,v in pairs(model_2:findModules(name)) do
--          local n = v.kW*v.kH*v.nOutputPlane
--          v.weight:normal(0,math.sqrt(2/n))
--          if cudnn.version >= 4000 then
--             v.bias = nil
--             v.gradBias = nil
--          else
--             v.bias:zero()
--          end
--       end
--    end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')

                                                         
    return model--, model_2
end

return createModel