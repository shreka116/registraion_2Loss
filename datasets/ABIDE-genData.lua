--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--  Downloads data from web and save them as .t7 extensions
--

local image = require 'image'
local paths = require 'paths'
local ffi = require 'ffi'

local M = {}

function M.exec(opt, cacheFile)
    if not paths.dir(opt.genData .. opt.dataset .. '/data') then
	    dofile('generateData.lua')	
    end    
    local tr_vl_split   = io.open(opt.genData .. opt.dataset .. '/brainMRI_train_val.txt','r')
    local dir           = paths.dir(opt.genData .. opt.dataset .. '/data/')
    -- print(opt.genData .. opt.dataset .. '/data/')
    table.sort(dir)

    -- local train_pair    = torch.CudaTensor(22232, 6, 384, 512)  -- previously counted
    -- local train_flow    = torch.CudaTensor(22232, 2, 384, 512)
    -- local val_pair      = torch.CudaTensor(640, 6, 384, 512)    -- previously counted
    -- local val_flow      = torch.CudaTensor(640, 2, 384, 512)
    local maxLength         = math.max(-1, #(opt.genData .. opt.dataset .. '/data/' .. 'brain_xxxxx_x.png')+1)
    local train_imagePath   = torch.CharTensor(21000, 2, maxLength)
    local train_gtPath      = torch.CharTensor(21000, maxLength)
    local val_imagePath     = torch.CharTensor(3000, 2, maxLength)
    local val_gtPath        = torch.CharTensor(3000, maxLength)

    local tr_cnt        = 1
    local vl_cnt        = 1
    local counter       = 1
    -- print(dir)
    for line in tr_vl_split:lines() do
        if line == '1' then     -- train data
        -- print(opt.genData .. opt.dataset .. '/data/' .. dir[counter*3 + 1])
        -- print(tr_cnt)
            ffi.copy(train_imagePath[{ {tr_cnt},{1},{} }]:data(), opt.genData .. opt.dataset .. '/data/' .. dir[counter*3])
            ffi.copy(train_imagePath[{ {tr_cnt},{2},{} }]:data(), opt.genData .. opt.dataset .. '/data/' .. dir[counter*3 + 1])
            ffi.copy(train_gtPath[{ {tr_cnt},{} }]:data(), opt.genData .. opt.dataset .. '/data/' .. dir[counter*3 + 2])

            -- train_pair[{ {tr_cnt},{1,3},{},{} }]    = image.load(opt.genData .. '/FlyingChairs_release/data/' .. dir[counter*3 + 1])  -- loading reference frame
            -- train_pair[{ {tr_cnt},{4,6},{},{} }]    = image.load(opt.genData .. '/FlyingChairs_release/data/' .. dir[counter*3 + 2])  -- loading target frame
            -- train_flow[{ {tr_cnt},{},{},{} }]       = readFlowFile(opt.genData .. '/FlyingChairs_release/data/' .. dir[counter*3])    -- loading ground-truth flow  

            tr_cnt = tr_cnt + 1
        elseif line == '2' then -- validation data
            ffi.copy(val_imagePath[{ {vl_cnt},{1},{} }]:data(), opt.genData .. opt.dataset .. '/data/' .. dir[counter*3])
            ffi.copy(val_imagePath[{ {vl_cnt},{2},{} }]:data(), opt.genData .. opt.dataset .. '/data/' .. dir[counter*3 + 1])
            ffi.copy(val_gtPath[{ {vl_cnt},{} }]:data(), opt.genData .. opt.dataset ..'/data/' .. dir[counter*3 + 2])

            -- val_pair[{ {vl_cnt},{1,3},{},{} }]      = image.load(opt.genData .. '/FlyingChairs_release/data/' .. dir[counter*3 + 1])
            -- val_pair[{ {vl_cnt},{4,6},{},{} }]      = image.load(opt.genData .. '/FlyingChairs_release/data/' .. dir[counter*3 + 2])
            -- val_flow[{ {vl_cnt},{},{},{} }]         = readFlowFile(opt.genData .. '/FlyingChairs_release/data/' .. dir[counter*3])

            vl_cnt = vl_cnt + 1
        end
        counter = counter + 1
    end

    local datasetInfo = {
        train   =   {
            imagePath   =   train_imagePath,
            targetPath  =   train_gtPath,
        },
        val     =   {
            imagePath   =   val_imagePath,
            targetPath  =   val_gtPath,
        },
    }

    print(" | saving list of ABIDE brainMRI dataset to " .. cacheFile)
    torch.save(cacheFile, datasetInfo)
    return datasetInfo    
end

return M
