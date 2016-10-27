--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

require 'image'

local M = {}


-- function M.SelectTransform(transforms)
--    return function(input)
--       local trTypes     = #transforms -- + 1           -- added one for identity (w/o transformation)
--       local numAugment  = torch.randperm(2)--torch.randperm(trTypes)
--       local selected    = torch.randperm(trTypes)
      
--         -- for typeNo, transform in ipairs(transforms) do

--       for augType = 1, numAugment[1] do    
           
--           input = transforms[selected[augType]](input)

--       end

--       return input
--    end
-- end


-- Randomly samples one of transforms to perform
function M.SelectTransform(transforms)
   return function(input)

        local photometric  = torch.random(1,5)
        local geometric    = torch.random(5,9)

        if #transforms ~= 1 then
          input = transforms[photometric](input)
          input = transforms[geometric](input)
        end
        
        -- input = transforms[#transforms](input)

      return input
   end
end


-- function M.Compose(transforms)
--    return function(input)
--       for idx, transform in ipairs(transforms) do
--         --  print(tostring(idx) .. '-->' .. tostring(transform))
--          input = transform(input)
--         --  print(tostring(idx) .. '-input -->' .. tostring(input:size()))
--       end
--     --   print('outa for loop-input -->' .. tostring(input:size()))
--       return input
--    end
-- end

function M.ColorNormalize(meanstd)
   return function(img)
      img = img:clone()
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
         img[i]:div(meanstd.std[i])
      end
      return img
   end
end

-- -- Scales the smaller edge to size
-- function M.Scale(size, interpolation)
--    interpolation = interpolation or 'bicubic'
--    return function(input)
--       local w, h = input:size(3), input:size(2)
--       if (w <= h and w == size) or (h <= w and h == size) then
--          return input
--       end
--       if w < h then
--          return image.scale(input, size, h/w * size, interpolation)
--       else
--          return image.scale(input, w/h * size, size, interpolation)
--       end
--    end
-- end

-- Crop to centered rectangle
function M.CenterCrop(size)
   return function(input)
      local w1 = math.ceil((input:size(3) - size)/2)
      local h1 = math.ceil((input:size(2) - size)/2)
      return image.crop(input, w1, h1, w1 + size, h1 + size) -- center patch
   end
end

-- Random crop form larger image with optional zero padding
function M.RandomCrop(size, padding)
   padding = padding or 0

   return function(input)
      if padding > 0 then
         local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
         temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
         input = temp
      end

      local w, h = input:size(3), input:size(2)
      if w == size and h == size then
         return input
      end

      local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
      local out = image.crop(input, x1, y1, x1 + size, y1 + size)
      assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
      return out
   end
end

-- Four corner patches and center crop from image and its horizontal reflection
function M.TenCrop(size)
   local centerCrop = M.CenterCrop(size)

   return function(input)
      local w, h = input:size(3), input:size(2)

      local output = {}
      for _, img in ipairs{input, image.hflip(input)} do
         table.insert(output, centerCrop(img))
         table.insert(output, image.crop(img, 0, 0, size, size))
         table.insert(output, image.crop(img, w-size, 0, w, size))
         table.insert(output, image.crop(img, 0, h-size, size, h))
         table.insert(output, image.crop(img, w-size, h-size, w, h))
      end

      -- View as mini-batch
      for i, img in ipairs(output) do
         output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
      end

      return input.cat(output, 1)
   end
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function M.RandomScale(minSize, maxSize)
   return function(input)
      local w, h = input:size(3), input:size(2)

      local targetSz = torch.random(minSize, maxSize)
      local targetW, targetH = targetSz, targetSz
      if w < h then
         targetH = torch.round(h / w * targetW)
      else
         targetW = torch.round(w / h * targetH)
      end

      return image.scale(input, targetW, targetH, 'bicubic')
   end
end

-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
function M.RandomSizedCrop(size)
   local scale = M.Scale(size)
   local crop = M.CenterCrop(size)

   return function(input)
      local attempt = 0
      repeat
         local area = input:size(2) * input:size(3)
         local targetArea = torch.uniform(0.08, 1.0) * area

         local aspectRatio = torch.uniform(3/4, 4/3)
         local w = torch.round(math.sqrt(targetArea * aspectRatio))
         local h = torch.round(math.sqrt(targetArea / aspectRatio))

         if torch.uniform() < 0.5 then
            w, h = h, w
         end

         if h <= input:size(2) and w <= input:size(3) then
            local y1 = torch.random(0, input:size(2) - h)
            local x1 = torch.random(0, input:size(3) - w)

            local out = image.crop(input, x1, y1, x1 + w, y1 + h)
            assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

            return image.scale(out, size, size, 'bicubic')
         end
         attempt = attempt + 1
      until attempt >= 10

      -- fallback
      return crop(scale(input))
   end
end



-- Lighting noise (AlexNet-style PCA-based noise)
function M.Lighting(alphastd, eigval, eigvec)
   return function(input)
      if alphastd == 0 then
         return input
      end

      local alpha = torch.Tensor(3):normal(0, alphastd)
      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()

      input = input:clone()
      for i=1,3 do
         input[i]:add(rgb[i])
      end
      return input
   end
end

local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function M.Saturation(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Brightness(var)
   local gs

   return function(input)
      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

-- function M.Contrast(var)
--    local gs

--    return function(input)
--       gs = gs or input.new()
--       grayscale(gs, input)
--       gs:fill(gs[1]:mean())

--       local alpha = 1.0 + torch.uniform(-var, var)
--       blend(input, gs, alpha)
--       return input
--    end
-- end

-- function M.RandomOrder(ts)
--    return function(input)
--       local img = input.img or input
--       local order = torch.randperm(#ts)
--       for i=1,#ts do
--          img = ts[order[i]](img)
--       end
--       return input
--    end
-- end

-- function M.ColorJitter(opt)
--    local brightness = opt.brightness or 0
--    local contrast = opt.contrast or 0
--    local saturation = opt.saturation or 0

--    local ts = {}
--    if brightness ~= 0 then
--       table.insert(ts, M.Brightness(brightness))
--    end
--    if contrast ~= 0 then
--       table.insert(ts, M.Contrast(contrast))
--    end
--    if saturation ~= 0 then
--       table.insert(ts, M.Saturation(saturation))
--    end

--    if #ts == 0 then
--       return function(input) return input end
--    end

--    return M.RandomOrder(ts)
-- end

function M.AdditiveGausNoise(var_1, var_2)

    return function(input)
        local gs = input.new()
        gs:resizeAs(input):zero()
        -- print('additive Gaussian Noise')
        -- print(gs:size())

        local sigma = torch.uniform(var_1, var_2)
        torch.normal(gs:select(1,1), 0, sigma)
            gs:select(1,2):copy(gs:select(1,1))
            -- gs:select(1,3):copy(gs:select(1,1))
            -- gs:select(1,4):copy(gs:select(1,1))
            -- gs:select(1,5):copy(gs:select(1,1))
            -- gs:select(1,6):copy(gs:select(1,1))
       
        return input:add(gs)
    end
end

function M.Contrast(var_1, var_2)

   return function(input)
      local gs = input.new()
      gs:resizeAs(input):zero()
      -- print('Contrast')
      -- print(gs:size())

    --   local ref_gray = rgb2gray(input[{ {1,3},{},{} }])
    --   local tar_gray = rgb2gray(input[{ {4,6},{},{} }])
      -- grayscale(gs[{ {1,3},{},{} }], input[{ {1,3},{},{} }])
      -- grayscale(gs[{ {4,6},{},{} }], input[{ {4,6},{},{} }])

      gs[{ {1},{},{} }]:fill(input[{ {1},{},{} }][1]:mean())
      gs[{ {2},{},{} }]:fill(input[{ {2},{},{} }][1]:mean())

      local alpha = 1.0 + torch.uniform(var_1, var_2)
      blend(input, gs, alpha)
      return input
   end
end

function M.MultiplicativeColorChange(var_1, var_2)

    return function(input)

      local mult_R = torch.uniform(var_1, var_2)
      local mult_G = torch.uniform(var_1, var_2)
      local mult_B = torch.uniform(var_1, var_2)

      input:select(1,1):mul(mult_R)
      input:select(1,2):mul(mult_G)
      input:select(1,3):mul(mult_B)
      input:select(1,4):mul(mult_R)
      input:select(1,5):mul(mult_G)
      input:select(1,6):mul(mult_B)


      return input
    end
end

function M.AdditiveBrightness(var)

    return function(input) 
      -- print('AdditiveBrightness')
      -- print(input:size())
      -- local ref_hsl = image.rgb2hsl(input[{ {1,3},{},{} }])
      -- local tar_hsl = image.rgb2hsl(input[{ {4,6},{},{} }])
      local changes = torch.normal(0, 0.2)
      -- ref_hsl:select(1,3):add(changes)
      -- tar_hsl:select(1,3):add(changes)
      -- input[{ {1,3},{},{} }]:copy(image.hsl2rgb(ref_hsl))
      -- input[{ {4,6},{},{} }]:copy(image.hsl2rgb(tar_hsl))

      input[{ {1},{},{} }]:add(changes)
      input[{ {2},{},{} }]:add(changes)
      return input
    end
end

function M.GammaChanges(var_1, var_2)

    return function(input) 
      -- print('GammaChanges')
      -- print(input:size())
      -- local ref_hsl = image.rgb2hsl(input[{ {1,3},{},{} }])
      -- local tar_hsl = image.rgb2hsl(input[{ {4,6},{},{} }])
      local gamma   = torch.uniform(var_1, var_2)
      -- ref_hsl:select(1,3):pow(gamma)
      -- tar_hsl:select(1,3):pow(gamma)
      -- input[{ {1,3},{},{} }]:copy(image.hsl2rgb(ref_hsl))
      -- input[{ {4,6},{},{} }]:copy(image.hsl2rgb(tar_hsl))
      input[{ {1},{},{} }]:pow(gamma)
      input[{ {2},{},{} }]:pow(gamma)
      return input
    end
end

function M.Translations(var)
    
    return function(input)

       local trans_x = torch.uniform(-var*input:size(3), var*input:size(3))
       local trans_y = torch.uniform(-var*input:size(3), var*input:size(3))

    --    local x_from, x_to, y_from, y_to = 0,0,0,0

    --    if (trans_x < 0) and (trans_y < 0) then
    --       x_from  = 1
    --       x_to    = inputSize[3] + trans_x
    --       y_from  = 1
    --       y_to    = inputSize[2] + trans_y
    --    elseif (trans_x < 0) and (trans_y >= 0) then
    --       x_from  = 1
    --       x_to    = inputSize[3] + trans_x
    --       y_from  = trans_y
    --       y_to    = inputSize[2]
    --    elseif (trans_x >= 0) and (trans_y < 0) then
    --       x_from  = trans_x
    --       x_to    = inputSize[3]
    --       y_from  = 1
    --       y_to    = inputSize[2] + trans_y
    --    elseif (trans_x >= 0) and (trans_y >= 0) then
    --       x_from  = trans_x
    --       x_to    = inputSize[3]
    --       y_from  = trans_y
    --       y_to    = inputSize[2]
    --    end

    --    local preDefinedSz = {256, 384}
    --    local rndx   = torch.random(x_from, x_to - preDefinedSz[2])
    --    local rndy   = torch.random(y_from, y_to - preDefinedSz[1])
    --    local rndx2  = torch.random(x_from, x_to - preDefinedSz[2])
    --    local rndy2  = torch.random(y_from, y_to - preDefinedSz[1])
    --     -- channels x 256 x 384 image
    --   --   print(x_from,x_to,y_from,y_to)
    --   --  print(rndx,rndy, rndx2, rndy2)


    --    local translated     = image.crop(input, rndx, rndy, rndx + preDefinedSz[2], rndy + preDefinedSz[1])

    -- --    print(input:size())
    -- --    print(gs:size())
       
    -- --    input[{ {1,3},{},{} }] = image.translate(gs[{ {1,3},{},{} }], trans_x, trans_y)
    -- --    input[{ {4,6},{},{} }] = image.translate(gs[{ {4,6},{},{} }], trans_x, trans_y)
       
       return image.translate(input, trans_x, trans_y)
      -- return translated
    end
end

function M.HorizontalFlip(prob)
   return function(input)
      if torch.uniform() < prob then
        --  input[{ {1,3},{},{} }] = image.hflip(input[{ {1,3},{},{} }])
        --  input[{ {4,6},{},{} }] = image.hflip(input[{ {4,6},{},{} }])

      end
      return image.hflip(input)
   end
end

function M.Rotation(var)
   return function(input)
      local deg = torch.uniform(-var, var)
      if deg ~= 0 then
        --  input[{ {1,3},{},{} }] = image.rotate(input[{ {1,3},{},{} }], deg * math.pi / 180, 'bilinear')
        --  input[{ {4,6},{},{} }] = image.rotate(input[{ {4,6},{},{} }], deg * math.pi / 180, 'bilinear')

      end
      return image.rotate(input, deg * math.pi / 180, 'bilinear')
   end
end

function M.Scales(minSize, maxSize)
   return function(input)
      local w, h        = input:size(3), input:size(2)
      local factors     = torch.uniform(minSize, maxSize)
      local w1          = math.ceil(w*factors)
      local h1          = math.ceil(h*factors)
      local scaled      = torch.zeros(input:size())

      if factors > 1 then
        local w_c = math.ceil(w1/2)
        local h_c = math.ceil(h1/2)
        local tmp_scled = image.scale(input, w1, h1)
        -- scaled[{ {},{},{} }] = tmp_scled[{ {},{h_c - math.floor(h/2), h_c + math.floor(h/2)},{w_c - math.floor(w/2), w_c + math.floor(w/2)} }]
        scaled[{ {},{},{} }] = tmp_scled[{ {},{h_c - 128, h_c + 127},{w_c - 128, w_c + 127} }]
      elseif factors < 1 then
        scaled[{ {},{1,h1},{1,w1} }] = image.scale(input, w1, h1)
      else
        scaled = input:clone()
      end

      
      return scaled--, rel_scaled_input  
   end
end

-- function M.Scales(w1, h1)
--    return function(input)
--       -- local w, h        = input:size(3), input:size(2)
--       -- local factors     = torch.uniform(minSize, maxSize)
--       -- local w1          = math.ceil(w*factors)
--       -- local h1          = math.ceil(h*factors)
--       local scaled      = image.scale(input, w1, h1)
      
--       return scaled--, rel_scaled_input  
--    end
-- end

function M.Identity()
   return function(input)
      return input
   end
end

return M
