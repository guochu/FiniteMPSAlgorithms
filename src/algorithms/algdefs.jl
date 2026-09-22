# algorithm configuration types (following TEMPO src/algorithms.jl)

abstract type MPSAlgorithm end
abstract type IterativeMPSAlgorithm <: MPSAlgorithm end

"""
	SVDCompression(; trunc=truncdimcutoff(D=Defaults.D, ϵ=Defaults.tol, add_back=0), verbosity=0)

Parameters of the SVD-sweep (one-pass) compression route: the exact product is formed first
and then compressed by an SVD right-orthogonalization sweep under the scheme `trunc`.
"""
@kwdef struct SVDCompression{T<:TruncationScheme} <: MPSAlgorithm
	trunc::T = truncdimcutoff(D = Defaults.D, ϵ = Defaults.tol, add_back = 0)
	verbosity::Int = 0
end

"""
	DMRG1(; maxiter=Defaults.maxiter, tol=Defaults.tol, D=Defaults.D, verbosity=0)

Parameters of the single-site variational (ALS) sweep engine: pure iteration parameters
(maximal sweeps, convergence tolerance, verbosity) and the bond
dimension `D` of the working guess: the out-of-place entry points draw their initial
guess with bond cap `alg.D` (`svdguess_add`, `svdguess_mult`, `svdguess_hadamard`,
`svdguess_compress`, `randommps` / `randommpo`), and the in-place variants re-fit a
caller-provided guess to `alg.D` bonds with `changebond!`. `DMRG1` itself does not
truncate.
"""
@kwdef struct DMRG1 <: IterativeMPSAlgorithm
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	D::Int = Defaults.D
	verbosity::Int = 0
end

const DefaultMultAlg = SVDCompression(trunc=DefaultTruncation)

"""
	iterative_compute!(cache, alg::DMRG1) -> khist

Repeat `sweep!(cache, alg)` until the relative difference of the LAST loss of two
successive sweeps satisfies `|lₙ - lₙ₋₁| / |lₙ₋₁| < alg.tol` (or `alg.maxiter` is
reached). The per-site losses of a sweep are monotone in processing time, so the last
value of a sweep is the best (most recently updated) loss and is comparable across
sweeps. Returns `khist`, the loss history of every sweep.

The sweeps keep the working data in mixed canonical form but never touch the Schmidt
values; the cache constructors therefore reset them (`unset_svectors!`) on the working
guess, so no stale spectrum can masquerade as a canonical gauge.
"""
function iterative_compute!(cache, alg; kwargs...)
	khist = Vector{Vector{Float64}}()
	prev = Inf
	for _ in 1:alg.maxiter
		kvals = sweep!(cache, alg; kwargs...)
		push!(khist, kvals)
		last = kvals[end]
		delta = isfinite(prev) ? (prev == 0 ? abs(last) : abs(last - prev) / abs(prev)) : Inf
		(alg.verbosity > 1) && println("DMRG iteration: delta = ", delta)
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
