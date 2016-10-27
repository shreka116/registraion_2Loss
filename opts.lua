--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--


local M = {}

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 Brain MRI Registration via CNN Training script')
--   cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
--   cmd:option('-data',       '',         'Path to dataset')
   cmd:option('-dataset',           'ABIDE',        'Options: brainMRI | ABIDE | toyShapes')
   cmd:option('-manualSeed',         0,             'Manually set RNG seed')
   cmd:option('-nGPU',               1,             'Number of GPUs to use by default')
   cmd:option('-backend',           'cudnn',        'Options: cudnn | cunn')
   cmd:option('-cudnn',             'fastest',      'Options: fastest | default | deterministic')
   cmd:option('-genData',        '../brainMRI/',    'Path to save generated files')
   ------------- Data options ------------------------
   cmd:option('-nThreads',          2,              'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',           100,            'Number of total epochs to run')
   cmd:option('-epochNumber',       1,              'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',         4,              'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',          'false',        'Run on validation set only')
--    cmd:option('-tenCrop',        'false', 'Ten-crop testing')
   ------------- Checkpointing options ---------------
   cmd:option('-save',              'checkpoints',  'Directory in which to save checkpoints')
   cmd:option('-resume',            'none',         'Resume from the latest checkpoint in this directory')
   cmd:option('-retrain',            'none',        'Resume from the latest checkpoint in this directory')
   ---------- Optimization options ----------------------
   cmd:option('-learningRate',      0.001,          'initial learning rate')
   cmd:option('-momentum',          0.9,            'momentum')
   cmd:option('-weightDecay',       5e-4,           'weight decay')
   cmd:option('-beta_1',            0.9,            'first parameter of Adam optimizer')
   cmd:option('-beta_2',           0.999,           'second parameter of Adam optimizer')
   cmd:option('-optimizer',         'sgd',          'Options: sgd | adagrad | adam')
      ---------- Model options ----------------------------------
   cmd:option('-networkType',       'SIMPLEnet',    'Options: USOFnet | EdgeUSOFnet')
   cmd:option('-optimState',        'none',         'Path to an optimState to reload from')
   ---------- Hyper parameters  ------------------------------
--   cmd:option('-epsilon',           0.001,      'Usually we use 0.001')
--   cmd:option('-photo_char',        0.25,       'Power of Chrabonnier Penalty, FlyingChairs = 0.25 , KITTI = 0.38')
--   cmd:option('-smooth_char',       0.37,       'Power of Chrabonnier Penalty, FlyingChairs = 0.37 , KITTI = 0.21')
--   cmd:option('-smooth_weight',       1,        'FlyingChairs = 1 , KITTI = 0.53')
--    cmd:option('-brainImgSize',      500,            'brain MRI image size')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   if opt.dataset == 'brainMRI' then
      -- Default nEpochs=90
      opt.nEpochs = opt.nEpochs == 0 and 200 or opt.nEpochs
   elseif opt.dataset == 'ABIDE' then
      -- Default nEpochs=90
      opt.nEpochs = opt.nEpochs == 0 and 200 or opt.nEpochs
   elseif opt.dataset == 'toyShapes' then
      -- Default nEpochs=90
      opt.nEpochs = opt.nEpochs == 0 and 200 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   return opt
end

return M
