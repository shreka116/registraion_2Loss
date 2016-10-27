--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--
--  Model creating code

require 'nn'
require 'cunn'
require 'cudnn'
require 'tvnorm-nn'
require 'stn'
require 'nngraph'

local M = {}

function M.setup(opt, checkpoint)
    local model
    -- local warp_image
    if checkpoint then
        local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
        -- local modelPath_2 = paths.concat(opt.resume, checkpoint.modelFile_2)
        assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
        print('=> Resuming model from ' .. modelPath)
        model     = torch.load(modelPath):cuda()
        -- assert(paths.filep(modelPath_2), 'Saved model not found: ' .. modelPath_2)
        -- print('=> Resuming model from ' .. modelPath_2)
        -- model_2   = torch.load(modelPath_2):cuda()
    elseif opt.retrain ~= 'none' then
        assert(paths.filep(modelPath), 'Model not found: ' .. opt.retrain)
        print('=> Loading model from ' .. opt.retrain)
        model   = torch.load(opt.retrain):cuda()
        --         assert(paths.filep(modelPath_2), 'Model not found: ' .. opt.retrain)
        -- print('=> Loading model from ' .. opt.retrain)
        -- model_2   = torch.load(opt.retrain):cuda()
    else
        print('=> Creating model from: models/' .. opt.networkType .. '.lua')
        model= require('models/' .. opt.networkType)(opt)
    end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
    --   model_2:apply(function(m)
    --      if m.setMode then m:setMode(1, 1, 1) end
    --   end)
   end    

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end

   local imgH   = 0
   local imgW   = 0

   if opt.dataset == "brainMRI" then
      imgH = 384
      imgW = 384
   end


   local criterion  = nn.MSECriterion():cuda()
   local criterion_2 = nn.MSECriterion():cuda()

   model:cuda()
   warp_image:cuda()
   cudnn.convert(model, cudnn)
   cudnn.convert(warp_image, cudnn)

   return model, warp_image, criterion, criterion_2

end

return M
