# algorithm configuration types (following TEMPO src/algorithms.jl)

abstract type MPSAlgorithm end
abstract type DMRGAlgorithm <: MPSAlgorithm end
abstract type TimeEvolutionAlgorithm <: MPSAlgorithm end

"""
	SVDCompression(; trunc=truncdimcutoff(D=Defaults.D, ϵ=Defaults.tol, add_back=0), verbosity=0)
	SVDCompression(trunc::TruncationScheme; verbosity=0)

Parameters of the SVD-sweep (one-pass) compression route: the exact product is formed first
and then compressed by an SVD right-orthogonalization sweep under the scheme `trunc`.
"""
struct SVDCompression{T<:TruncationScheme} <: DMRGAlgorithm
	trunc::T
	verbosity::Int
end
SVDCompression(trunc::TruncationScheme; verbosity::Int=0) = SVDCompression(trunc, verbosity)
SVDCompression(; trunc::TruncationScheme=truncdimcutoff(D=Defaults.D, ϵ=Defaults.tol, add_back=0), verbosity::Int=0) =
	SVDCompression(trunc, verbosity)
Base.similar(x::SVDCompression; trunc::TruncationScheme=x.trunc, verbosity::Int=x.verbosity) =
	SVDCompression(trunc; verbosity=verbosity)

# truncation schemes carrying an explicit maximum bond dimension `D` seed the
# bond-dimension cap of the initial guesses (`svdguess_*`, `increase_bond!`)
const TruncationWithD = Union{TruncateDim, TruncateDimCutoff}

"""
	DMRG1(; maxiter=Defaults.maxiter, tol=Defaults.tol, verbosity=0, callback=Returns(nothing))

Parameters of the single-site variational (ALS) sweep engine: pure iteration parameters
(maximal sweeps, convergence tolerance, verbosity, per-sweep callback). `DMRG1` does
not seed initial guesses and does not truncate; the initial guess — and with it the
bond-dimension cap — is supplied by the caller, e.g. `svdguess_add`, `svdguess_mult`,
`svdguess_hadamard`, `svdguess_compress`, `increase_bond!`, `randommps` / `randommpo`.
"""
struct DMRG1 <: DMRGAlgorithm
	maxiter::Int
	tol::Float64
	verbosity::Int
	callback::Function
end
function DMRG1(; maxiter::Int=Defaults.maxiter, tol::Float64=Defaults.tol,
			   verbosity::Int=0, callback::Function=Returns(nothing))
	return DMRG1(maxiter, tol, verbosity, callback)
end
function Base.similar(x::DMRG1; maxiter::Int=x.maxiter, tol::Float64=x.tol,
					  verbosity::Int=x.verbosity, callback=x.callback)
	return DMRG1(; maxiter, tol, verbosity, callback)
end

const DefaultMultAlg = SVDCompression(DefaultTruncation)

"""
	iterative_compute!(cache, alg::DMRG1) -> khist

Repeat `sweep!(cache, alg)` until the relative difference of the LAST loss of two
successive sweeps satisfies `|lₙ - lₙ₋₁| / |lₙ₋₁| < alg.tol` (or `alg.maxiter` is
reached). The per-site losses of a sweep are monotone in processing time, so the last
value of a sweep is the best (most recently updated) loss and is comparable across
sweeps. Returns `khist`, the loss history of every sweep.
"""
function iterative_compute!(cache, alg::DMRG1)
	khist = Vector{Vector{Float64}}()
	prev = Inf
	for _ in 1:alg.maxiter
		kvals = sweep!(cache, alg)
		push!(khist, kvals)
		alg.callback(cache, kvals)
		last = kvals[end]
		delta = isfinite(prev) ? (prev == 0 ? abs(last) : abs(last - prev) / abs(prev)) : Inf
		(alg.verbosity > 1) && println("DMRG1 iteration: delta = ", delta)
		delta < alg.tol && break
		prev = last
	end
	return khist
end

# final relative difference of the last losses of the last two sweeps — the convergence
# measure of `iterative_compute!`, evaluated a posteriori from the loss history
function _iterative_delta(khist)
	length(khist) < 2 && return Inf
	last = khist[end][end]
	prev = khist[end-1][end]
	return prev == 0 ? abs(last) : abs(last - prev) / abs(prev)
end
