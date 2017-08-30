require 'math'
--[=====[

    Copyright (C) 2015 John-Alexander Assael (www.johnassael.com)
    https://github.com/iassael/torch7-decomposition

    The MIT License (MIT)

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to do
    so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

--]=====]

local differential_entropy = function(x)
    -- Center data
    local mean = torch.mean(x, 1)
    local x_m = x - torch.ger(torch.ones(x:size(1)), mean:squeeze())

    -- Calculate Covariance
    local cov = torch.mm(x_m, x_m:t())
    cov:div(x:size(2) - 1)

    -- Get eigenvalues and eigenvectors
    local ce = torch.symeig(cov)
    local orig_ce = ce:clone()
    ce:abs():add(1e-5)

    -- Return differential entropy of the resulting transformed distribution,
    -- which is sum(log(sqrt(variance * 2 * pi * e))) = sum(log(variance) / 2 + log(2  pi * e)
    local result = ce:mul(2 * math.exp(1) * math.pi):log():div(2):sum()

    if result ~= result then
      print('We have a wonky result.')
      print('The original ce vector was:')
      print(ce)
    end

    return result
end

local pca = function(x, n_comp, whitened, normalize)
    -- Center data
    local mean = torch.mean(x, 1)
    local x_m = x - torch.ger(torch.ones(x:size(1)), mean:squeeze())

    -- Calculate Covariance
    local cov = torch.mm(x_m, x_m:t())
    cov:div(x:size(2) - 1)

    -- Get eigenvalues and eigenvectors
    local ce, cv = torch.symeig(cov, 'V')

    -- Sort eigenvalues
    local ce, idx = torch.sort(ce, true)

    -- Sort eigenvectors
    cv = cv:index(2, idx:long())

    -- Keep only the top
    if n_comp and n_comp < cv:size(2) then
        ce = ce:sub(1, n_comp)
        cv = cv:sub(1, -1, 1, n_comp)
    end

    local u = nil

    if normalize then
      -- Check if whitened version
      -- vectors are divided by the singular values to
      -- ensure uncorrelated outputs with unit component-wise variances.
      if not whitened then
          ce:add(1e-5):sqrt()
      end

      -- Get inverse
      local inv_ce = ce:clone():pow(-1)

      -- Make it a matrix with diagonal inv_ce
      local inv_diag = torch.diag(inv_ce)

      -- Transform to get U
      u = torch.mm(torch.mm(x_m:t(), cv), inv_diag)
    else
      -- Transform to get U without normalization
      u = torch.mm(x_m:t(), cv)
    end

    return u
end

print(differential_entropy)

return {pca, differential_entropy}
