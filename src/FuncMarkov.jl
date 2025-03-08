module FuncMarkov
# ==============================================================================
import LinearAlgebra: eigen
import Statistics

using StaticArrays, NamedArrays
import StatsFuns: normcdf


# ------------------------------------------------------------------------------
export tauchen, tauchen_log
export FunctionMarkovChain


# ------------------------------------------------------------------------------
# alias: std elementary type
const F64 = Float64
const F32 = Float32
const I64 = Int64
const Str = String
const Sym = Symbol

# alias: std collections
const V64 = Vector{Float64}
const M64 = Matrix{Float64}
const V32 = Vector{Float32}
const M32 = Matrix{Float32}
const Dict64 = Dict{Symbol,Float64}
const NTup64{D} = NTuple{D,Float64}

# alias: std abstract types
const AbsV = AbstractVector
const AbsM = AbstractMatrix
const AbsVM = AbstractVecOrMat

const Iterable{D} = Union{AbsV{D}, Tuple{Vararg{D}}}

# alias: StaticArrays.jl
const SV64{D}   = SVector{D,Float64}
const SM64{D,K} = SMatrix{D,K,Float64}

# alias: NamedArrays.jl
const NmV64 = NamedVector{Float64}
const NmM64 = NamedMatrix{Float64}










#=******************************************************************************
TAUCHEN 1986
******************************************************************************=#
"""
    tauchen(
        N::Int, 
        ρ::Real, 
        σ::Real; 
        yMean::Real = 0.0, 
    	nσ   ::Real = 3,
    )::@NamedTuple{states::V64, probs::M64}

Discretize an AR(1) process with mean `xMean`, persistence `ρ`, and standard
deviation `σ` into `N` states. The function returns a Markov chain with the
states and the transition probabilities.

`y_t = (1-ρ)*xMean + ρ*y_{t-1} + σ*ϵ_t, ϵ_t ~ N(0,1)`

The argument `nσ` is the number of standard deviations to include in the
discretization. The chosen states include the endpoints.

## Reference
This function is a modified version of:
`https://github.com/hendri54/shared/blob/master/%2Bar1LH/tauchen.m`
"""
function tauchen(
    N     ::Int,
    ρ     ::Real,
    σ     ::Real;
    yMean ::Real = 0.0,
    nσ    ::Real = 3,
)::@NamedTuple{states::V64, probs::M64}
    @assert N > 1 "N must be > 1"
    @assert σ > 0 "σ must be > 0"
    @assert nσ > 0 "nσ must be > 0"

    # Width of grid
    a_bar = nσ * sqrt(σ^2.0 / (1.0 - ρ^2))

    # Grid
    y = LinRange(-a_bar, a_bar, N)

    # Distance between points
    d = y[2] - y[1]

    # get transition probabilities
    trProbM = zeros(N, N)
    for iRow in 1:N
        # do end points first
        trProbM[iRow,1] = normcdf((y[1] - ρ*y[iRow] + d/2) / σ)
        trProbM[iRow,N] = 1 - normcdf((y[N] - ρ*y[iRow] - d/2) / σ)

        # fill the middle columns
        for iCol = 2:N-1

            trProbM[iRow,iCol] = (
                normcdf((y[iCol] - ρ*y[iRow] + d/2) / σ) -
                normcdf((y[iCol] - ρ*y[iRow] - d/2) / σ)
            )

        end # iCol
    end # iRow

    # normalize the probs to rowsum = 1 due to possible float errors
    trProbM ./= sum(trProbM, dims=2)

    # don't forget to shift the process to the position of the long-term mean
    return (
        states = y .+ yMean,
        probs  = trProbM
    )
end # tauchen
# ------------------------------------------------------------------------------
"""
    tauchen_log(
        N::Int, 
        ρ::Real,
        σ::Real;
        logyMean::Real = 0.0, 
        nσ::Real = 3
    )::@NamedTuple{states::V64, probs::M64}

Tauchen 1986 but for log-normal AR(1) process:

`log(y_{t}) = (1-ρ)*logyMean + ρ*log(y_{t-1}) + σ*ϵ_t, ϵ_t ~ N(0,1)`

The function returns a Markov chain with the states and the transition
probabilities.
"""
function tauchen_log(
    N        ::Int,
    ρ        ::Real,
    σ        ::Real;
    logyMean ::Real = 0.0,
    nσ       ::Real = 3,
)::@NamedTuple{states::V64, probs::M64}
    mc_log = tauchen(N, ρ, σ; yMean=logyMean, nσ=nσ)
    return (
        states = exp.(mc_log.states),
        probs  = mc_log.probs
    )
end # tauchen_log




#=******************************************************************************
MAIN STRUCTURE
******************************************************************************=#
"""
    FunctionMarkovChain{K,D,T}

A K-state Markov chain in which each state is a D-dimensional scalar function:

`f_k(x) : R^D -> Any`

The transition matrix is a KxK matrix of functions:

`P[i,j] = Pr{f_j(x) | f_i(x)}`

The states are allowed to be labeled by anything: symbol, string, integer, or
even composite types (e.g. tuple of numbers). The label's type is `T`.

## Fields
- `names::SizedVector{K,T}`: the labels of the states
- `funs ::SizedVector{K,Function}`: the functions
- `Pr   ::SizedMatrix{K,K,F64}`: the transition matrix


## Constructor

    FunctionMarkovChain(
        funs::Iterable{K,Function},
        d   ::Int,
        Pr  ::AbsM ;
        names::Union{Iterable,Nothing} = nothing
    )

where `funs` is a vector or a tuple of `K` functions, `d` is the dimension of
the functions, `Pr` is the transition matrix, and `names` is an optional vector
or tuple of `K` labels for the states.

If no `names` are provided, the states are labeled by Symbols: `:f1`,`:f2`, etc.



## Example
```julia
fmv = include("FuncMarkov.jl")

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


# test: overloaded Base methods

size(mc) # returns (D,K)

mc[1]    # returns the first function
mc[:f1]  # returns the first function (by name)

mc[:f1](rand(2)) # evaluate the first function at a random point




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

"""
mutable struct FunctionMarkovChain{D,K,T}
    names::SizedVector{K,T}
    funs ::SizedVector{K,Function}
    Pr   ::SizedMatrix{K,K,F64}

    function FunctionMarkovChain(
        funs::Iterable{Function},
        d   ::Int,
        Pr  ::AbsM ;
        names::Union{Iterable,Nothing} = nothing
    )
        k = length(funs)
        @assert size(Pr) == (k,k) "The transition matrix must be KxK square"
        @assert d >= 0 "The dimension of the functions must be non-negative"
        @assert all(
            isapprox.(
                sum(Pr, dims = 2), 
                1, 
                atol = 1E-6
            )
        ) "The rows of the transition matrix must sum to 1 at atol = 1E-6"

        nameType = isnothing(names) ? Symbol : eltype(names)
        _names = if isnothing(names)
            SizedVector{k,Symbol}([Symbol("f$i") for i in 1:k])
        else
            @assert length(names) == k "The number of names must be equal to K"
            SizedVector{k,nameType}(names...)
        end
        new{d,k,nameType}(
            _names,
            SizedVector{k,Function}(funs),
            SizedMatrix{k,k,F64}(Pr)
        )
    end # constructor
end # FunctionMarkovChain
# ------------------------------------------------------------------------------
function Base.show(io::IO, mc::FunctionMarkovChain{D,K,T}) where {D,K,T}
    println(io, "FunctionMarkovChain{", length(mc.funs), "}")
    println(io, "- f(x), x ∈ R^", D)
    println(io, "- label type       : ", T)
    println(io, "- # function states: ", K)
    
    _prettymat = NamedArray(
        mc.Pr,
        names    = (mc.names, mc.names),
        dimnames = ("from", "to")
    )
    println(io, "- transition matrix:")
    display(_prettymat)

    return nothing
end






# ------------------------------------------------------------------------------
function Base.size(mc::FunctionMarkovChain{D,K,T}) where {D,K,T}
    return (D,K)
end
# ------------------------------------------------------------------------------
function Base.getindex(mc::FunctionMarkovChain{D,K,T}, i::Int) where {D,K,T}
    @assert 1 <= i <= K "Index out of bounds"
    return mc.funs[i]
end
function Base.getindex(mc::FunctionMarkovChain{D,K,T}, name::T) where {D,K,T}
    i = findfirst(mc.names .== name)
    @assert !isnothing(i) "State of label $name not found"
    return mc.funs[i]
end






# ------------------------------------------------------------------------------
"""
    stationary(mc::FunctionMarkovChain{D,K,T})::V64

Computes the stationary distribution of the Markov chain. The function returns
a vector of probabilities.
"""
function stationary(mc::FunctionMarkovChain{D,K,T})::V64 where {D,K,T}
    λ, V = eigen(mc.Pr')
    i    = argmin(abs.(λ .- 1))
    pss  = V[:,i] ./ sum(V[:,i])
    return pss .|> real
end





# ------------------------------------------------------------------------------
"""
    mean(
        mc::FunctionMarkovChain{D,K,T}, 
        x ::AbsV, 
        i ::Int
    )

Computes the conditional expectation of the function `f(x)` given today's 
state being the `i`-th state.
"""
function Statistics.mean(
    mc::FunctionMarkovChain{D,K,T}, 
    x ::AbsV, 
    i ::Int
) where {D,K,T}
    @assert 1 <= i <= K "Index out of bounds"
    return sum([f(x) for f in mc.funs] .* mc.Pr[i,:])
end
# ------------------------------------------------------------------------------
"""
    mean(
        mc::FunctionMarkovChain{D,K,T}, 
        x ::AbsV, 
        name ::T
    )

Computes the conditional expectation of the function `f(x)` given today's 
state labelled by `name`.
"""
function Statistics.mean(
    mc   ::FunctionMarkovChain{D,K,T}, 
    x    ::AbsV, 
    name ::T
) where {D,K,T}
    i = findfirst(mc.names .== name)
    @assert !isnothing(i) "State of label $name not found"
    return sum([f(x) for f in mc.funs] .* mc.Pr[i,:])
end





































# ==============================================================================
end # module FuncMarkov