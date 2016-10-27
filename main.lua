--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--

require 'gnuplot'
require 'torch'
require 'paths'
require 'optim'
-- require 'hzproc'
require 'nn'

local models        = require 'models/init'
local DataLoader    = require 'dataloader'
local opts          = require 'opts'
local Trainer       = require 'train'
local checkpoints   = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
cutorch.setDevice(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, warp_image, criterion , criterion_2 =  models.setup(opt, checkpoint)
-- print(model)
-- print(warp_image)
-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)


-- for n, sample in valLoader:run() do
--     print(n)
--     print(sample)
-- end

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, warp_image, criterion, criterion_2, opt, optimState)

--------TO DO -----------TO DO----------------
if opt.testOnly then
   local loss = trainer:test(0, valLoader)
   print(string.format(' * Results loss: %1.4f  top5: %6.3f', loss))
   return
end
---------------------------------------------


--------TO DO -----------TO DO----------------
-- local trainLosses = {}
-- local testLosses = {}

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local Losses     = checkpoint and torch.load('checkpoints/Losses_' .. startEpoch-1 .. '.t7') or {trainLosses = {}, testLosses = {}}

for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainLoss  = trainer:train(epoch, trainLoader)
   Losses.trainLosses[#Losses.trainLosses + 1] = trainLoss
   gnuplot.pngfigure('losses/trainLoss.png')
   gnuplot.plot({torch.range(1, #Losses.trainLosses), torch.Tensor(Losses.trainLosses), '-'})
   gnuplot.plotflush()
   -- Run model on validation set
--    local EPE_below, EPE_above, EPE_all, testLoss   = trainer:test(epoch, valLoader)
   local testLoss   = trainer:test(epoch, valLoader)
   Losses.testLosses[#Losses.testLosses + 1] = testLoss
   gnuplot.pngfigure('losses/testLoss.png')
   gnuplot.plot({torch.range(1, #Losses.testLosses), torch.Tensor(Losses.testLosses), '-'})
   gnuplot.plotflush()


   if (epoch%10 == 0) then
        checkpoints.save(epoch, model, trainer.optimState, opt)
        torch.save('checkpoints/Losses_' .. epoch .. '.t7', Losses)

   end
--    checkpoints.save(epoch, model, warp_image, trainer.optimState, opt)
end
---------------------------------------------
