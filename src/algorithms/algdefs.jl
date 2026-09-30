# algorithm configuration types (following TEMPO src/algorithms.jl)

abstract type MPSAlgorithm end
abstract type IterativeMPSAlgorithm <: MPSAlgorithm end

"""
	SingleSiteUpdate

Abstract base of the single-site (ALS) sweep engines: `DMRG1`, `ALSLinSolve`,
`ALSRecon` and `Seq2Seq`.
"""
abstract type SingleSiteUpdate <: IterativeMPSAlgorithm end

"""
	TwoSiteUpdate

Abstract base of the two-site sweep engines: `DMRG2` (the concrete `ALSLinSolve2`),
`CADMRG` and `PDMRG`.
"""
abstract type TwoSiteUpdate <: IterativeMPSAlgorithm end

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
	DMRG1(; maxiter=Defaults.maxiter, tol=Defaults.tol, D=Defaults.D,
	      alg_eigsolve=Defaults.alg_eigsolve(), alg_orth=Defaults.alg_orth(), verbosity=0)

Parameters of the single-site variational (ALS) sweep engine: pure iteration parameters
(maximal sweeps, convergence tolerance, verbosity) and the bond
dimension `D` of the working guess: the out-of-place entry points draw their initial
guess with bond cap `alg.D` (`svdguess_add`, `svdguess_mult`, `svdguess_hadamard`,
`svdguess_compress`, `randommps` / `randommpo`), and the in-place variants re-fit a
caller-provided guess to `alg.D` bonds with `changebond!`. `DMRG1` itself does not
truncate.

The local problems are configured by the two solver fields: `alg_eigsolve` (a
`KrylovKit` algorithm object, see [`Defaults.alg_eigsolve`](@ref)) drives the local
effective-Hamiltonian `eigsolve`, and `alg_orth` (a tensor factorization such as
`QRpos()`, see [`Defaults.alg_orth`](@ref)) the QR/LQ gauge moves — the right-to-left
moves use the adjoint `alg_orth'`.
"""
@kwdef struct DMRG1{E,O} <: SingleSiteUpdate
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	D::Int = Defaults.D
	alg_eigsolve::E = Defaults.alg_eigsolve()
	alg_orth::O = Defaults.alg_orth()
	verbosity::Int = 0
end

"""
	DMRG2(; maxiter=Defaults.maxiter, tol=Defaults.tol, trunc=Defaults.alg_trunc(),
	      alg_eigsolve=Defaults.alg_eigsolve(), alg_orth=Defaults.alg_orth(), verbosity=0)

Parameters of the two-site variational sweep engine: the neighboring site pair is
optimized jointly and re-split by an SVD under `trunc::TruncationScheme`, so the bond
dimension adapts during the sweeps (growth where the environment demands it, truncation
where the scheme caps it). All engines accepting a `DMRG1` (`mult`, `add`, `compress`,
`hadamard`, `linsolve`, `ground_state`) accept a `DMRG2` as well; the dedicated
two-site `linsolve` configuration is [`ALSLinSolve2`](@ref). `alg.trunc` is also
consulted for the bond cap of automatically drawn initial guesses
(`_truncation_bond`).

`alg_eigsolve` (a `KrylovKit` algorithm object, see [`Defaults.alg_eigsolve`](@ref))
drives the local effective-Hamiltonian `eigsolve`. `alg_orth`
([`Defaults.alg_orth`](@ref)) is accepted for interface uniformity with `DMRG1`; the
two-site re-splits are truncating SVDs under `alg.trunc` (factorization `SDD`) and
carry the gauge themselves, so no QR/LQ moves consume it.
"""
@kwdef struct DMRG2{TR<:TruncationScheme,E,O} <: TwoSiteUpdate
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	trunc::TR = Defaults.alg_trunc()
	alg_eigsolve::E = Defaults.alg_eigsolve()
	alg_orth::O = Defaults.alg_orth()
	verbosity::Int = 0
end

"""
	ALSLinSolve(; maxiter=Defaults.maxiter, tol=Defaults.tol, D=Defaults.D,
	            alg_linsolve=Defaults.alg_linsolve(), verbosity=0)

Algorithm configuration of the iterative `linsolve`: single-site ALS sweeps over the
normal-equation stacks with the bond profile `alg.D`; the local normal equations are
solved by the KrylovKit iterative solver `alg.alg_linsolve` (matrix-free, the current
site tensor as warm start).
"""
@kwdef struct ALSLinSolve <: SingleSiteUpdate
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	D::Int = Defaults.D
	alg_linsolve::KrylovKit.LinearSolver = Defaults.alg_linsolve()
	verbosity::Int = 0
end

"""
	ALSLinSolve2(; maxiter=Defaults.maxiter, tol=Defaults.tol, trunc=Defaults.alg_trunc(),
	             alg_linsolve=Defaults.alg_linsolve(), verbosity=0)

Algorithm configuration of the two-site iterative `linsolve`: two-site ALS sweeps over
the normal-equation stacks with the bond profile `alg.trunc`; the local normal equations
are solved by the KrylovKit iterative solver `alg.alg_linsolve` (matrix-free, the current
site-pair tensor as warm start).
"""
@kwdef struct ALSLinSolve2{TR<:TruncationScheme, S<:KrylovKit.LinearSolver} <: TwoSiteUpdate
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	trunc::TR = Defaults.alg_trunc()
	alg_linsolve::S = Defaults.alg_linsolve()
	verbosity::Int = 0
end

# the bond cap carried by a truncation scheme (`nothing` for schemes without one)
_truncation_bond(t::TruncateDim) = t.D
_truncation_bond(t::TruncateDimCutoff) = t.D
_truncation_bond(t::TruncationScheme) = nothing

# the bond cap for drawing automatic initial guesses from a truncation scheme
# (`Defaults.D` for schemes without a bond cap)
_guess_bond(t::TruncationScheme) = something(_truncation_bond(t), Defaults.D)

const DefaultMultAlg = SVDCompression(trunc=Defaults.alg_trunc())

"""
	ALSConvergenceInfo(niter, converged, losses, itererr, residual)
	ALSConvergenceInfo(losses, converged; residual=nothing)

Convergence report appended as the last return value of every `IterativeMPSAlgorithm`
entry point:

* `niter`: number of sweeps performed;
* `converged`: whether `iterative_compute!` reached `alg.tol`;
* `losses`: the per-sweep loss history (what `iterative_compute!` used to return);
* `itererr`: the final relative loss difference (the convergence measure);
* `residual`: the problem's residual when it is exactly the loss (set by `linsolve`,
  `seq2seq` and `reconstruct`, whose loss *is* the residual), `nothing` for the other
  drivers (there the loss is the residual up to an unknown constant or not directly
  the residual at all).
"""
struct ALSConvergenceInfo
	niter::Int
	converged::Bool
	losses::Vector{Vector{Float64}}
	itererr::Float64
	residual::Union{Float64, Nothing}
end

function ALSConvergenceInfo(losses::Vector{Vector{Float64}}, converged::Bool; residual::Union{Float64, Nothing} = nothing)
	niter = length(losses)
	itererr = _iterative_delta(losses)
	return ALSConvergenceInfo(niter, converged, losses, itererr, residual)
end

"""
	iterative_compute!(cache, alg; residual=false, kwargs...) -> info::ALSConvergenceInfo

Repeat `sweep!(cache, alg)` until the relative difference of the LAST loss of two
successive sweeps satisfies `|lₙ - lₙ₋₁| / |lₙ₋₁| < alg.tol` (or `alg.maxiter` is
reached). The per-site losses of a sweep are monotone in processing time, so the last
value of a sweep is the best (most recently updated) loss and is comparable across
sweeps. Returns `info`, the [`ALSConvergenceInfo`](@ref) carrying the per-sweep loss
history and the convergence report; with `residual = true` the `residual` field is set
to the final loss (for the drivers whose loss *is* the problem's residual — `linsolve`,
`seq2seq`, `reconstruct`). Remaining `kwargs` are forwarded to `sweep!`.

The sweeps keep the working data in mixed canonical form but never touch the Schmidt
values; the cache constructors therefore reset them (`unset_svectors!`) on the working
guess, so no stale spectrum can masquerade as a canonical gauge.

Logging follows the package-wide `verbosity` convention (header of
`FiniteMPSAlgorithms.jl`): level ≥ 2 summarizes the final (best) loss of every sweep
(`_logiter`); level ≥ 3 additionally prints the loss of every local update from within
the sweeps, tagged with the half-sweep direction (`_logupdate`).
"""
function iterative_compute!(cache, alg; residual::Bool = false, kwargs...)
	khist = Vector{Vector{Float64}}()
	prev = Inf
	delta = Inf
	converged = false
	for iter in 1:alg.maxiter
		kvals = sweep!(cache, alg; kwargs...)
		push!(khist, kvals)
		last = kvals[end]
		delta = isfinite(prev) ? (prev == 0 ? abs(last) : abs(last - prev) / abs(prev)) : Inf
		(alg.verbosity > 1) && _logiter(stdout, _algname(alg), iter, delta, "loss" => last)
		if delta < alg.tol
			converged = true
			break
		end
		prev = last
	end
	(!converged && alg.verbosity > 0) && @warn "iterative_compute! reached maxiter without " *
		"converging: $(alg.maxiter) sweeps, final delta = $(round(delta; sigdigits=4)), tol = $(alg.tol)"
	return ALSConvergenceInfo(khist, converged; residual = residual ? khist[end][end] : nothing)
end

# iteration logging (following InfiniteMPSAlgorithms `_logiter`)
function _logiter(io::IO, name::AbstractString, iter::Int, err::Real, extra::Pair...)
	str = join(["$k = $(repr(round(v; sigdigits = 8)))" for (k, v) in extra], ", ")
	@printf(io, "%s iter %4d : ϵ = %.3e %s\n", name, iter, err, str)
	return nothing
end

# per-local-update loss logging inside the sweeps (`leftsweep!` / `rightsweep!` call
# this right after each local optimization), tagged with the half-sweep direction
function _logupdate(io::IO, dir::AbstractString, i::Int, k::Real)
	@printf(io, "%s site %3d : loss = %.6e\n", dir, i, k)
	return nothing
end

# the per-algorithm name for the iteration log (`DMRG2{...}` -> "DMRG2")
_algname(alg) = string(typeof(alg).name.name)

# final relative difference of the last losses of the last two sweeps — the convergence
# measure of `iterative_compute!`, evaluated a posteriori from the loss history
function _iterative_delta(khist)
	length(khist) < 2 && return Inf
	last = khist[end][end]
	prev = khist[end-1][end]
	return prev == 0 ? abs(last) : abs(last - prev) / abs(prev)
end
