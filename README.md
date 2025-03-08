# FuncMarkov.jl
Implements a special kind of Markov chains where each state is a function. This is also known to be an infinite-dimensional Markov chain, or Markov chain in functional space (e.g. Banach space)



## Installation

```julia
add "https://github.com/Clpr/FuncMarkov.jl.git#main"
```


## Usage


```julia
import FuncMarkov as fmv


fs = [
    x -> x[1] + x[2],
    x -> x[1] - x[2]
]


# test: construct from an AR(1) by Tauchen (1986)
fNames, P = fmv.tauchen(2, 0.9, 0.1, nσ = 0.5) # AR(1)
fNames, P = fmv.tauchen(2, 0.9, 0.1, nσ = 0.5) # log-AR(1)
mc = fmv.FunctionMarkovChain(fs, 2, P, names=fNames)

# test: change the names, functions, and Pr
mc.names .= [0.0, 1.0]

# test: standard manual verbose constructor
P  = [0.9 0.1; 0.2 0.8]
mc = fmv.FunctionMarkovChain(fs, 2, P, names = [:f1, :f2])

# test: a human-friendly display
display(mc)

# test: overloaded Base methods

size(mc) # returns (D,K)

mc[1]    # returns the first function
mc[:f1]  # returns the first function (by name)

mc[:f1](rand(2)) # evaluate the first state function at a random point


# test: overloaded standard library functions

fmv.stationary(mc) # stationary distribution

# NOTE: `mean()` works only if `sum()` and scalar product are defined for the 
# function return's type. I expect this package to be used with numerical 
# functions which returns numeric values which almost always satisfy the above 
# conditions.

import Statistics
Statistics.mean(mc, rand(2), 1)   # E{f(x)|k = 1}, the conditional expectation.
Statistics.mean(mc, rand(2), :f1) # index by name



# test: non-scalar returns

fs = [
    x -> sin.(x),
    x -> cos.(x)
]
P  = [0.9 0.1; 0.2 0.8]
mc = fmv.FunctionMarkovChain(fs, 2, P)
Statistics.mean(mc, rand(2), 1)  # returns a mean array 
```

Use `varinfo(FuncMarkov)` to see what else functions are exported.
Every exported function or type has self-contained docstrings.


## License 

MIT license


