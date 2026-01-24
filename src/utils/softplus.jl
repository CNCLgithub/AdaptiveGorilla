# This code was copied from https://github.com/JuliaStats/LogExpFunctions.jl/blob/62b081dad466f3b74f3cf8fb7e51472b1ed7024a/src/basicfuns.jl#L160
# Their licence is posted below
#
#
#The LogExpFunctions.jl package is licensed under the MIT "Expat" License:

# Original work Copyright (c) 2015: Dahua Lin, StatsFuns.jl contributors.
# Modified version Copyright (c) 2019: Tamas K. Papp.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function softplus(x::Float64)
    x1, x2, x3, x4 = (-745.1332191019412, -36.7368005696771, 18.021826694558577, 33.23111882352963)
    if x < x1
        return zero(x)
    elseif x < x2
        return exp(x)
    elseif x < x3
        return log1p(exp(x))
    elseif x < x4
        return x + exp(-x)
    else
        return x
    end
end

