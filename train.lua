--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('BRR.Trainer', M)

function Trainer:__init(model, warp_image, criterion, criterion_2, opt, optimState)
   self.model           = model
   self.warp_image         = warp_image
  --  print(warp_image)
   self.criterion       = criterion
   self.criterion_2     = criterion_2
   self.optimState      = optimState or {
      learningRate      = opt.learningRate,
      learningRateDecay = 0.0,
      momentum          = opt.momentum,
      nesterov          = true,
      dampening         = 0.0,
      weightDecay       = opt.weightDecay,
      beta1             = opt.beta_1,
      beta2             = opt.beta_2,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
  --  self.params_2, self.gradParams_2 = warp_image:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)
   local timer              = torch.Timer()
   local dataTimer          = torch.Timer()
   local loss,loss_2 = 0.0,0.0

   local function feval()
    --   return self.criterion.output, self.gradParams
      -- return (self.criterion.output+self.criterion_2.output)/2, self.gradParams
      return (loss +loss_2), self.gradParams
   end
  --     local function feval_2()
  --   --   return self.criterion.output, self.gradParams
  --     return self.criterion_2.output, self.gradParams_2
  --  end

   local losses     = {}
   local trainSize  = dataloader:size()
   local lossSum    = 0.0
   local debug_loss = 0.0
   local N          = 0

   print('=============================')
   print(self.optimState)
   print('=============================')
   
   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
  --  self.warp_image:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output      = self.model:forward(self.input)
      -- output[{ {},{1},{3} }] = output[{ {},{1},{3} }]/128
      -- output[{ {},{2},{3} }] = output[{ {},{2},{3} }]/128

      -- local output_2    = self.warp_image:forward{self.input[{ {},{1},{},{} }] , self.model.output}
      local output_2    = self.warp_image:forward{self.input[{ {},{1},{},{} }] , output}

      local batchSize   = output:size(1)
           
       loss        = self.criterion:forward(self.model.output, self.target)
       loss_2      = self.criterion_2:forward(self.warp_image.output, self.input[{ {},{2},{},{} }])

      self.model:zeroGradParameters()
      -- self.warp_image:zeroGradParameters()

      self.criterion:backward(self.model.output, self.target)
      self.criterion_2:backward(self.warp_image.output, self.input[{ {},{2},{},{} }])


      local gradLoss = self.warp_image:backward({self.input[{ {},{1},{},{} }] , self.model.output}, self.criterion_2.gradInput)

      self.model:backward(self.input, (self.criterion.gradInput:add(gradLoss[2])))

      local garbage, tmp_loss, tmp_loss_2 = 0,0,0
      if self.opt.optimizer == 'sgd' then
        garbage, tmp_loss = optim.sgd(feval, (self.params), self.optimState)
        -- garbage, tmp_loss_2 = optim.sgd(feval, self.params_2, self.optimState)
      elseif self.opt.optimizer == 'adam' then
        garbage, tmp_loss = optim.adam(feval, (self.params), self.optimState)
        -- garbage, tmp_loss_2 = optim.adam(feval, self.params_2, self.optimState)
      elseif self.opt.optimizer == 'adagrad' then
        garbage, tmp_loss = optim.adagrad(feval, (self.params), self.optimState)
        -- garbage, tmp_loss_2 = optim.adagrad(feval, self.params_2, self.optimState)
      end

      N = n
      lossSum = lossSum + (loss +loss_2)
      debug_loss = debug_loss + (loss + loss_2)

      if (n%200) == 0 then
          -- print(string.format('Gradient min: %1.4f \t max:  %1.4f \t norm: %1.4f', torch.min(self.gradParams:float()), torch.max(self.gradParams:float()), torch.norm(self.gradParams:float())))
          print(string.format('Gradient min: %1.4f \t max:  %1.4f \t norm: %1.4f', torch.min(self.gradParams:float()), torch.max(self.gradParams:float()), torch.norm(self.gradParams:float())))
    
          image.save('losses/current_ref.png', self.input[{ {1},{1},{},{} }]:reshape(1,self.input[{ {1},{1},{},{} }]:size(3),self.input[{ {1},{1},{},{} }]:size(4)))
          image.save('losses/current_tar.png', self.input[{ {1},{2},{},{} }]:reshape(1,self.input[{ {1},{2},{},{} }]:size(3),self.input[{ {1},{2},{},{} }]:size(4)))
          image.save('losses/estimate_tar.png', self.warp_image.output[{ {1},{},{},{} }]:reshape(1,self.warp_image.output[{ {1},{},{},{} }]:size(3),self.warp_image.output[{ {1},{},{},{} }]:size(4)))
          -- print(self.target:size())
          -- print(output:size())
          print('============================================================================')
          print(('|  Predicted Matrix: [%2.4f  %2.4f  %2.4f]     GroundTruth Matrix: [%2.4f  %2.4f  %2.4f]'):format(output[1][1][1],output[1][1][2],output[1][1][3],self.target[1][1][1],self.target[1][1][2],self.target[1][1][3]))
          print(('|                    [%2.4f  %2.4f  %2.4f]                         [%2.4f  %2.4f  %2.4f]'):format(output[1][2][1],output[1][2][2],output[1][2][3],self.target[1][2][1],self.target[1][2][2],self.target[1][2][3]))
          print('============================================================================')
          -- collectgarbage()
      end

	    if (n%50) == 0 then
   	     print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  loss %1.4f'):format(
             epoch, n, trainSize, timer:time().real, dataTime, loss))--total_loss))
   	   -- check that the storage didn't get changed due to an unfortunate getParameters call
   	  end

 	    assert(self.params:storage() == self.model:parameters()[1]:storage())
      --  assert(self.params_2:storage() == self.warp_image:parameters()[1]:storage())
      timer:reset()
      dataTimer:reset()
   end

--    return top1Sum / N, top5Sum / N, lossSum / N
    return lossSum / N
end

function Trainer:test(epoch, dataloader)

   local timer    = torch.Timer()
   local size     = dataloader:size()
   local N        = 0
   local glob_n   = 0
   local lossSum  = 0.0
   local debug_loss = 0.0
   local losses   = {}
   local param_11, param_12, param_13, param_21, param_22, param_23 = 0.0,0.0,0.0,0.0,0.0,0.0
   
   self.model:evaluate()
  --  self.warp_image:evaluate()

   for n, sample in dataloader:run() do
      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output      = self.model:forward(self.input)
      local output_2    = self.warp_image:forward{self.input[{ {},{1},{},{} }] , self.model.output}

      local batchSize   = output:size(1)
           
      local loss        = self.criterion:forward(self.model.output, self.target)
      local loss_2      = self.criterion_2:forward(self.warp_image.output, self.input[{ {},{2},{},{} }])


      -- local batchSize   = output:size(1)
      -- local loss        = self.criterion:forward(self.model.output, self.target)

      -- N           = N + batchSize
      lossSum     = lossSum + (loss + loss_2)
      debug_loss  = debug_loss + (loss + loss_2)
      if (n%100) == 0 then
        print((' | Test: [%d][%d/%d]    Time %.3f  loss %1.4f'):format( epoch, n, size, timer:time().real, (loss + loss_2)))
  

        -- local mat = torch.zeros(3, 3):cuda()
        -- mat[{ {1,2},{} }] = output:select(1,1):clone()
        -- mat[3][3] = 1
        -- local test_img = hzproc.Transform.Fast(self.input[{ {1},{1},{},{} }]:reshape(1,256,256), mat:transpose(1,2));
        -- local test_img
        image.save('losses/test_ref.png', self.input[{ {1},{1},{},{} }]:reshape(1,self.input[{ {1},{1},{},{} }]:size(3),self.input[{ {1},{1},{},{} }]:size(4)))
        image.save('losses/test_tar.png', self.input[{ {1},{2},{},{} }]:reshape(1,self.input[{ {1},{2},{},{} }]:size(3),self.input[{ {1},{2},{},{} }]:size(4)))
        image.save('losses/testEst_tar.png', output_2[{ {1},{},{},{} }]:reshape(1,256,256))
      end

      timer:reset()
      glob_n = n
   end
   self.model:training()
  --  self.warp_image:training()

   return lossSum / glob_n
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if (epoch >= 200) and (epoch%30 == 0) then
	return self.optimState.learningRate/2
   else
	return self.optimState.learningRate
   end 

end

return M.Trainer
