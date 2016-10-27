by default : type "th main.lua"

    ------------ General options --------------------
   cmd:option('-dataset',           'ABIDE',        'Options: brainMRI | ABIDE')
   cmd:option('-manualSeed',         0,             'Manually set RNG seed')
   cmd:option('-nGPU',               1,             'Number of GPUs to use by default')
   cmd:option('-backend',           'cudnn',        'Options: cudnn | cunn')
   cmd:option('-cudnn',             'fastest',      'Options: fastest | default | deterministic')
   cmd:option('-genData',            '../brainMRI/',         'Path to save generated files')
   ------------- Data options ------------------------
   cmd:option('-nThreads',          2,              'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',           100,            'Number of total epochs to run')
   cmd:option('-epochNumber',       1,              'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',         4,              'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',          'false',        'Run on validation set only')
   ------------- Checkpointing options ---------------
   cmd:option('-save',              'checkpoints',  'Directory in which to save checkpoints')
   cmd:option('-resume',            'none',         'Resume from the latest checkpoint in this directory')
   cmd:option('-retrain',            'none',        'Resume from the latest checkpoint in this directory')
   ---------- Optimization options ----------------------
   cmd:option('-learningRate',      0.01,           'initial learning rate')
   cmd:option('-momentum',          0.9,            'momentum')
   cmd:option('-weightDecay',       5e-4,           'weight decay')
   cmd:option('-beta_1',            0.9,            'first parameter of Adam optimizer')
   cmd:option('-beta_2',           0.999,           'second parameter of Adam optimizer')
   cmd:option('-optimizer',         'sgd',          'Options: sgd | adagrad | adam')
      ---------- Model options ----------------------------------
   cmd:option('-networkType',       'SIMPLEnet',      'Options: USOFnet | EdgeUSOFnet')
   cmd:option('-optimState',        'none',         'Path to an optimState to reload from')
 

