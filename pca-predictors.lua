--[[

Compute all of the linear regressions between the given descriptions and their MSEs.

]]
require 'cutorch'
require 'nn'
require 'cunn'
require 'json'
require 'optim'
pca = require 'pca'

-- Normalize a sample to have mean 0
function normalize_mean(X)
  local N = X:clone()

  local w, h = N:size(1), N:size(2)

  -- First, get units normalized to mean 0 and standard deviation 1
  local M = N:sum(1):mul(1 / w)
  N:csub(
    M:view(1, h):expand(w, h)
  )

  return N
end

-- Normalize a sample to have standard deviation 1
function normalize_stdev(N)
  local Q = N:clone()

  local w, h = N:size(1), N:size(2)

  local S = Q:clone():pow(2):sum(1):mul(1 / w):sqrt()
  Q:cdiv(
    S:view(1, h):expand(w, h)
  )

  return Q
end

function train_linear_predictor(A, B)

  -- Split A and B in to training and test sets
  testA = A:narrow(1, 1, math.floor(A:size(1) / 2))
  A = A:narrow(1, math.floor(A:size(1) / 2), math.ceil(A:size(1) / 2))
  testB = B:narrow(1, 1, math.floor(B:size(1) / 2))
  B = B:narrow(1, math.floor(B:size(1) / 2), math.ceil(B:size(1) / 2))

  -- Analytic solution
  local transform_matrix = torch.gels(B, A)

  -- Mean squared error
  return (torch.mm(testA, transform_matrix) - testB):pow(2):mean(1)
end

function covariance_matrix(A, B) -- samples x sizeA, samples x sizeB
  -- Assume these are normalized, so return:
  return torch.mm(A:t(), B):mul(1 / A:size(1)) -- sizeA x sizeB
end

function main()
  cmd = torch.CmdLine()

  cmd:option('-desc_list', 'desc_list', 'File containing a list of description files (one per line)')
  cmd:option('-out_file', 'out_file', 'Output file')

  local opt = cmd:parse(arg)

  assert(path.exists(opt.desc_list), 'Description file list does not exist.')

  print('Reading out description file.')

  -- Read out lines of this file
  local filenames = {}
  local desc_file = io.open(opt.desc_list)
  while true do
    local line = desc_file:read("*line")
    if line == nil then break end
    table.insert(filenames, line)
  end
  io.close()

  print('Done.')

  -- For each file, get its described encoding,
  -- and extract sentence encodings, and normalize them
  -- to have standard deviation 1 and mean 0.
  local encodings = {}
  local indices_used = {}
  for i=1,#filenames do
    local filename = filenames[i]
    print('Reading out:', filename)
    local data = torch.load(filename)
    local all_embeddings = data['encodings']
    local sample_length = data['sample_length']

    -- for nc usage
    --[[
    local all_embeddings = torch.load(filename)
    local sample_length = 0

    for i=1,#all_embeddings do
      sample_length = sample_length + all_embeddings[i]:size(1)
    end
    ]]

    print('Done.')
    print('Normalizing:', filename)

    --local sample_length = #all_embeddings
    local rnn_size = all_embeddings[1]:size(2)

    local new_encoding = torch.Tensor(sample_length, rnn_size):zero()

    local k = 1
    for i=1,#all_embeddings do
      local vector = all_embeddings[i]

      for j=1,vector:size(1) do
        local element = vector[j]
        new_encoding[k] = nn.utils.recursiveType(element, 'torch.DoubleTensor')
        k = k + 1
      end
    end

    -- Normalize and make independent
    local normalized_new_encoding = pca(normalize_stdev(normalize_mean(new_encoding)):t(), 500, true, false)

    encodings[filename] = normalized_new_encoding
  end

  -- Now create the entire LSLR MSE table.
  comparison_table = {}
  for name_A, encoding_A in pairs(encodings) do
    mapping_table = {}
    for name_B, encoding_B in pairs(encodings) do
      if name_A ~= name_B then
        print('COMPARING:', name_A, name_B)

        -- Train nonlinear predictor
        local mse = train_linear_predictor(
          encoding_A,
          encoding_B
        )

        print(mse)

        -- Extract MSE into a Lua table
        -- and put it into the big mapping table
        mapping_table[name_B] = mse:totable()
      end
    end
    comparison_table[name_A] = mapping_table
  end

  -- Save the measurements as JSON, for
  -- visualization by a JS frontend.
  print('Writing...')
  local out_file = io.open(opt.out_file, 'w')
  out_file:write(json.encode(comparison_table))
  out_file:close()
  print('Done.')

end

main()
