require 'cutorch'
require 'hzproc'
require 'image'
require 'math'
require 'paths'
require 'nn'
require 'cunn'
require 'cudnn'

cutorch.setDevice(1)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')



local folderName    = '../brainMRI/ABIDE/'
local targetFolder  = '../brainMRI/ABIDE/data/'
local imgDIR        = paths.dir(folderName)
table.sort(imgDIR)

local targetImgSize = 256
local total_iter    = 24000
local img_cnt = 1
local ref_cnt = 1
print(imgDIR)
local ref       = image.load(folderName .. imgDIR[ref_cnt + 2])

paths.mkdir(targetFolder)

print('generating data ......')

local transform_model   = nn.Sequential()
local parnet            = nn.ParallelTable()
local matrix_generator  = nn.AffineTransformMatrixGenerator(true, true, true):cuda()
local trans             = nn.Sequential()

trans:add(nn.Identity())
trans:add(nn.Transpose({2,3},{3,4}))
parnet:add(trans)
parnet:add(nn.AffineGridGeneratorBHWD(256,256))
transform_model:add(parnet)
transform_model:add(nn.BilinearSamplerBHWD())
transform_model:add(nn.Transpose({3,4},{2,3}))

transform_model:cuda()
cudnn.convert(transform_model, cudnn)

for iter = 1, total_iter do
    if (iter%1000) == 0 and iter < 24000 then
        ref_cnt = ref_cnt + 1
        ref     = image.load(folderName .. imgDIR[ref_cnt + 2])
    end

    if (iter%1000) == 1 then
        local tmp_ref = image.scale(ref:float(), targetImgSize, targetImgSize)
        image.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_1.png', tmp_ref)

        local trans_x = torch.uniform(-20, 20)/128
        local trans_y = torch.uniform(-10, 10)/128
        local scale_x = torch.uniform(0.8, 1.2)
        local scale_y = torch.uniform(0.8, 1.2)
        local shear_x = torch.uniform(-20, 20)/128
        local shear_y = torch.uniform(-20, 20)/128
        local rotation= torch.uniform(-20, 20)*math.pi/180

        local tform = torch.CudaTensor(2,3):fill(0)
        tform[1] = torch.CudaTensor{(scale_x*math.cos(rotation)) - (scale_y*shear_y*math.sin(rotation)) , (scale_x*shear_x*math.cos(rotation)) - (scale_y*math.sin(rotation)) , trans_x*128 }
        tform[2] = torch.CudaTensor{(scale_x*math.sin(rotation)) + (scale_y*shear_y*math.cos(rotation)) , (scale_x*shear_x*math.sin(rotation)) + (scale_y*math.cos(rotation)) , trans_y*128 }

        local tmp_tar = transform_model:forward{tmp_ref:reshape(1,1,256,256):cuda() , tform:reshape(1,2,3):cuda()}:reshape(1,256,256)

        local tar = image.scale(tmp_tar:float(), targetImgSize, targetImgSize)
        image.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_2.png', tar)
        torch.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_gt.t7', tform)

        print('apply trasformations to image# ' .. tostring(ref_cnt) .. '...')

    else

        local trans_x = torch.uniform(-10, 10)/128
        local trans_y = torch.uniform(-10, 10)/128
        local scale_x = torch.uniform(0.9, 1.1)
        local scale_y = torch.uniform(0.9, 1.1)
        local shear_x = torch.uniform(-10, 10)/128
        local shear_y = torch.uniform(-10, 10)/128
        local rotation= torch.uniform(-10, 10)*math.pi/180


        local tform = torch.CudaTensor(2,3):fill(0)
        tform[1] = torch.CudaTensor{(scale_x*math.cos(rotation)) - (scale_y*shear_y*math.sin(rotation)) , (scale_x*shear_x*math.cos(rotation)) - (scale_y*math.sin(rotation)) , trans_x*128 }
        tform[2] = torch.CudaTensor{(scale_x*math.sin(rotation)) + (scale_y*shear_y*math.cos(rotation)) , (scale_x*shear_x*math.sin(rotation)) + (scale_y*math.cos(rotation)) , trans_y*128 }

        local tmp_ref = transform_model:forward{ref:reshape(1,1,256,256):cuda() , tform:reshape(1,2,3):cuda()}:reshape(1,256,256)

        local tmp_ref2= image.scale(tmp_ref:float(), targetImgSize, targetImgSize)
        image.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_1.png', tmp_ref2)

        -- ====================================
        -- ====================================

         trans_x = torch.uniform(-10, 10)/128
         trans_y = torch.uniform(-10, 10)/128
         scale_x = torch.uniform(0.9, 1.1)
         scale_y = torch.uniform(0.9, 1.1)
         shear_x = torch.uniform(-10, 10)/128
         shear_y = torch.uniform(-10, 10)/128
         rotation= torch.uniform(-10, 10)*math.pi/180

        tform = torch.CudaTensor(2,3):fill(0)
        tform[1] = torch.CudaTensor{(scale_x*math.cos(rotation)) - (scale_y*shear_y*math.sin(rotation)) , (scale_x*shear_x*math.cos(rotation)) - (scale_y*math.sin(rotation)) , trans_x*128 }
        tform[2] = torch.CudaTensor{(scale_x*math.sin(rotation)) + (scale_y*shear_y*math.cos(rotation)) , (scale_x*shear_x*math.sin(rotation)) + (scale_y*math.cos(rotation)) , trans_y*128 }

        local tmp_tar = transform_model:forward{tmp_ref:reshape(1,1,256,256):cuda() , tform:reshape(1,2,3):cuda()}:reshape(1,256,256)

        local tar = image.scale(tmp_tar:float(), targetImgSize, targetImgSize)

        image.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_2.png', tar)
        torch.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_gt.t7', tform)

    end

    img_cnt = img_cnt + 1
end

print('generating data DONE !!!! ')
print('========================================================')
 



-- function hzproc_Affine(img, trans_x, trans_y, scale_x, scale_y, shear_x, shear_y, rotation)
-- 	-- affine transformation matrix
-- 	mat = hzproc.Affine.Shift(trans_x, trans_y)
-- 	mat = mat * hzproc.Affine.Scale(scale_x, scale_y)
-- 	mat = mat * hzproc.Affine.Rotate(rotation)
-- 	mat = mat * hzproc.Affine.Shear(shear_x, shear_y)
-- 	-- affine mapping
-- 	outs = hzproc.Transform.Fast(img:cuda(), mat);
-- 	-- display the images
-- 	-- image.display(O)
--     return outs, mat:transpose(1,2)
-- end

-- for iter = 1, total_iter do

--     if (iter%1000) == 0 and iter < 24000 then
--         ref_cnt = ref_cnt + 1
--         ref     = image.load(folderName .. imgDIR[ref_cnt + 2])
--     end

--     if (iter%1000) == 1 then

--         local tmp_ref = image.scale(ref:float(), targetImgSize, targetImgSize)
--         image.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_1.png', tmp_ref)

--         local trans_x = torch.uniform(-20, 20)
--         local trans_y = torch.uniform(-10, 10)
--         local scale_x = torch.uniform(0.8, 1.2)
--         local scale_y = torch.uniform(0.8, 1.2)
--         local shear_x = torch.uniform(-0.2,0.2)
--         local shear_y = torch.uniform(-0.2,0.2)
--         local rotation= torch.uniform(-20, 20)*math.pi/180

--         local tform = torch.zeros(2,3)
--         -- tform[1][1]   = scale_x*math.cos(rotation)
--         -- tform[1][2]   = -shear_x*math.sin(rotation)
--         -- tform[1][3]   = trans_x
--         -- tform[2][1]   = scale_y*math.sin(rotation)
--         -- tform[2][2]   = shear_y*math.cos(rotation)
--         -- tform[2][3]   = trans_y

--         local tmp_tar,matt = hzproc_Affine(tmp_ref, trans_x, trans_y, scale_x, scale_y, shear_x, shear_y, rotation)
--         tform[1][1]      = matt[1][1]
--         tform[1][2]      = matt[1][2]
--         tform[1][3]      = matt[1][3]
--         tform[2][1]      = matt[2][1]
--         tform[2][2]      = matt[2][2]
--         tform[2][3]      = matt[2][3]


--         local tar     = image.scale(tmp_tar:float(), targetImgSize, targetImgSize)
--         image.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_2.png', tar)
--         torch.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_gt.t7', tform)

--         print('apply trasformations to image# ' .. tostring(ref_cnt) .. '...')
--     else

--         local trans_x = torch.uniform(-10, 10)
--         local trans_y = torch.uniform(-10, 10)
--         local scale_x = torch.uniform(0.9, 1.1)
--         local scale_y = torch.uniform(0.9, 1.1)
--         local shear_x = torch.uniform(-0.1,0.1)
--         local shear_y = torch.uniform(-0.1,0.1)
--         local rotation= torch.uniform(-10, 10)*math.pi/180

--         local tmp_ref = hzproc_Affine(ref, trans_x, trans_y, scale_x, scale_y, shear_x, shear_y, rotation)
--         local tmp_ref2= image.scale(tmp_ref:float(), targetImgSize, targetImgSize)
--         image.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_1.png', tmp_ref2)

--         -- ====================================
--         -- ====================================

--          trans_x = torch.uniform(-10, 10)
--          trans_y = torch.uniform(-10, 10)
--          scale_x = torch.uniform(0.9, 1.1)
--          scale_y = torch.uniform(0.9, 1.1)
--          shear_x = torch.uniform(-0.1,0.1)
--          shear_y = torch.uniform(-0.1,0.1)
--          rotation= torch.uniform(-10, 10)*math.pi/180

--         local tform = torch.zeros(2,3)
--         -- tform[1][1]   = scale_x*math.cos(rotation)
--         -- tform[1][2]   = -shear_x*math.sin(rotation)
--         -- tform[1][3]   = trans_x
--         -- tform[2][1]   = scale_y*math.sin(rotation)
--         -- tform[2][2]   = shear_y*math.cos(rotation)
--         -- tform[2][3]   = trans_y
 

--         local tmp_tar, matt = hzproc_Affine(tmp_ref2, trans_x, trans_y, scale_x, scale_y, shear_x, shear_y, rotation)
--         tform[1][1]      = matt[1][1]
--         tform[1][2]      = matt[1][2]
--         tform[1][3]      = matt[1][3]
--         tform[2][1]      = matt[2][1]
--         tform[2][2]      = matt[2][2]
--         tform[2][3]      = matt[2][3]

--         local tar     = image.scale(tmp_tar:float(), targetImgSize, targetImgSize)
--         image.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_2.png', tar)
--         torch.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_gt.t7', tform)

--     end

--     img_cnt = img_cnt + 1
-- end

-- print('generating data DONE !!!! ')
-- print('========================================================')
