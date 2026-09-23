# reconstruct: sample-amplitude MPS reconstruction by quadratic (least-squares)
# optimization — the variational alternative to tensor cross interpolation
# (plan/07_ls_reconstruction.md). Given samples {(𝐱⁽ᵏ⁾, aₖ)} of the amplitudes of a
# high-dimensional tensor (aₖ ∈ ℂ, possibly noisy), find a finite-bond MPS ψ
# minimizing the data objective
#   ℒ(ψ) = Σₖ |⟨𝐱⁽ᵏ⁾|ψ⟩ − aₖ|²
# by single-site ALS sweeps with a fixed bond profile alg.D (the DMRG1 discipline).
# Three environment structures are swept, mirroring seq2seq.jl:
#   Lmat[s], Rmat[s]: per-sample left/right contexts ℓ_s(𝐱ₖ), r_s(𝐱ₖ)  (the feature
#                     vector of sample k at site s is fₖ = ℓₖ ⊗ e(xₖ) ⊗ rₖ)
#   gstorage[s]:      ⟨ψ|ψ⟩_HS ridge transfer (the seq2seq regularization; added to
#                     the local normal equations only, not part of the reported loss)
# The local normal equation (M_s + α·R_s)·z = b_s is solved iteratively with
# KrylovKit's `linsolve` (matrix-free, the current site tensor as warm start); the
# gauge is moved by QR/LQ. Every per-site loss is the exact global data objective at
# that point of the sweep (monotone for exact solves). No KKT scale restoration: the
# normal equations are inhomogeneous, the data fixes the scale.

"""
	ALSRecon(; maxiter=Defaults.maxiter, tol=Defaults.tol, D=Defaults.D,
	        α=0.01, nadd=8, nbuffer=1024, verbosity=0)

Sample-amplitude MPS reconstruction by quadratic (least-squares) optimization —
the variational alternative to tensor cross interpolation. Given the sample set
`{(𝐱⁽ᵏ⁾, aₖ)}`, the fitted objective is the data loss

	ℒ(ψ) = Σₖ |⟨𝐱⁽ᵏ⁾|ψ⟩ − aₖ|²

where `⟨𝐱⁽ᵏ⁾|ψ⟩` is the wave-function amplitude of `ψ` on the basis product
`|𝐱⁽ᵏ⁾⟩ = ⊗ⱼ|xⱼ⁽ᵏ⁾⟩` and `aₖ ∈ ℂ` the measured amplitude (possibly noisy).

Single-site ALS sweeps with a fixed bond profile `alg.D` (the DMRG1 discipline: the
out-of-place entry points draw a random guess of bond `alg.D`, the in-place route
re-fits the caller's guess with `changebond!`). `α ≥ 0` is the Hilbert-Schmidt ridge
added to the local normal equations for conditioning (the seq2seq regularization; not
part of the reported loss); `nadd`/`nbuffer` drive the adaptive
sample-enrichment loop of [`reconstruct`](@ref).
"""
@kwdef struct ALSRecon <: IterativeMPSAlgorithm
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	D::Int = Defaults.D
	α::Float64 = 0.01        # HS ridge on the local solves (seq2seq regularizer)
	nadd::Int = 8            # samples added per adaptive enrichment round
	nbuffer::Int = 1024      # random candidate pool of the residual evaluation
	verbosity::Int = 0
end

# ---------- sample bookkeeping ----------

# samples: Vector of Pair{NTuple{L,Int},T} or (𝐱, a) tuples → (X::Matrix{L×N}, a::Vector)
function _normalize_samples(ds::NTuple{L,Int}, samples) where {L}
	N = length(samples)
	N > 0 || throw(ArgumentError("empty sample set"))
	X = Matrix{Int}(undef, L, N)
	vals = [smp isa Pair ? last(smp) : smp[2] for smp in samples]
	T = promote_type(eltype(vals), Float64)
	a = Vector{T}(undef, N)
	for (k, smp) in enumerate(samples)
		𝐱 = smp isa Pair ? first(smp) : smp[1]
		length(𝐱) == L || throw(DimensionMismatch("sample coordinates must have length $L"))
		for j in 1:L
			1 <= 𝐱[j] <= ds[j] ||
				throw(ArgumentError("sample coordinate out of range: site $j value $(𝐱[j])"))
			X[j, k] = 𝐱[j]
		end
		a[k] = vals[k]
	end
	return X, a
end

# ---------- per-sample context transfers ----------

# left context transfer: row k of the output is `C[k, :] * A[:, xₖ, :]` (N×Dl → N×Dr)
function _left_context_transfer(C::AbstractMatrix{T}, A::AbstractArray{T,3},
								xcol::AbstractVector{Int}) where {T}
	N = size(C, 1)
	out = zeros(T, N, size(A, 3))
	for p in 1:size(A, 2)
		for k in 1:N
			xcol[k] == p || continue
			@views out[k, :] .= vec(transpose(C[k, :]) * A[:, p, :])
		end
	end
	return out
end

# right context transfer: row k of the output is `A[:, xₖ, :] * C[k, :]` (N×Dr → N×Dl)
function _right_context_transfer(C::AbstractMatrix{T}, A::AbstractArray{T,3},
								 xcol::AbstractVector{Int}) where {T}
	N = size(C, 1)
	out = zeros(T, N, size(A, 1))
	for p in 1:size(A, 2)
		for k in 1:N
			xcol[k] == p || continue
			@views out[k, :] .= A[:, p, :] * C[k, :]
		end
	end
	return out
end

# ---------- ALS cache ----------

"""
	ALSReconCache: ALS problem carrier of [`reconstruct`](@ref) — the working chain `ψ`
	(raw data convention: `scaling == 1`, the represented amplitudes are fitted),
	the sample coordinates `X` (L×N) and target amplitudes `a`, the per-sample context
	stacks `Lmat`/`Rmat` and the HS ridge stack `gstorage` (seq2seq's third stack,
	rank-3 version).
"""
struct ALSReconCache{CP,T}
	ψ::CP
	X::Matrix{Int}
	a::Vector{T}
	Lmat::Vector{Matrix{T}}
	Rmat::Vector{Matrix{T}}
	gstorage::Vector{Matrix{T}}
end

function ALSReconCache(ψ::CanonicalMPS, X::Matrix{Int}, a::Vector)
	# fold the per-site scaling into the data (bounded per site): the sweeps match the
	# REPRESENTED amplitudes, and the result keeps `scaling == 1`
	s = scaling(ψ)
	if !isone(s)
		for A in ψ.data
			A .*= s
		end
		setscaling!(ψ, one(s))
	end
	# the sweeps never touch Schmidt values: reset them so "initialized" implies
	# "properly canonical" (package-wide discipline)
	unset_svectors!(ψ)
	L = length(ψ)
	N = length(a)
	T = promote_type(scalartype(ψ), eltype(a))
	ac = convert(Vector{T}, a)
	Lmat = Vector{Matrix{T}}(undef, L)
	Rmat = Vector{Matrix{T}}(undef, L)
	gstorage = Vector{Matrix{T}}(undef, L + 1)
	Lmat[1] = ones(T, N, 1)
	Rmat[L] = ones(T, N, 1)
	gstorage[L+1] = ones(T, 1, 1)
	c = ALSReconCache(ψ, X, ac, Lmat, Rmat, gstorage)
	_init_contexts_right!(c)
	return c
end

# right-to-left initialization of the right contexts and the ridge stack: Rmat[s]
# covers sites s+1..L, gstorage[s] is the HS overlap over sites s..L; Lmat[1] is the
# trivial boundary (the left contexts are transferred during the sweeps)
function _init_contexts_right!(c::ALSReconCache)
	L = length(c.ψ)
	for s in L:-1:1
		s < L && (c.Rmat[s] = _right_context_transfer(c.Rmat[s+1], c.ψ[s+1], @view(c.X[s+1, :])))
		c.gstorage[s] = _g_transfer_right(c.gstorage[s+1], c.ψ[s])
	end
	return c
end

# sweep-time transfers: ONE stack entry per site (the next site's left context /
# left ridge); the right contexts of the not-yet-reached sites stay untouched
# (the dmrg2.jl double-update lesson)
_left_transfer!(c::ALSReconCache, s) =
	(c.Lmat[s+1] = _left_context_transfer(c.Lmat[s], c.ψ[s], @view(c.X[s, :])); c)
_right_transfer!(c::ALSReconCache, s) =
	(c.Rmat[s-1] = _right_context_transfer(c.Rmat[s], c.ψ[s], @view(c.X[s, :])); c)
_g_transfer_left!(c::ALSReconCache, s) =
	(c.gstorage[s+1] = _g_transfer_left(c.gstorage[s], c.ψ[s]); c)
_g_transfer_right!(c::ALSReconCache, s) =
	(c.gstorage[s] = _g_transfer_right(c.gstorage[s+1], c.ψ[s]); c)

# ---------- ridge transfers (rank-3 versions of seq2seq's _g_updateleft/_updateright) ----------

function _g_transfer_left(g::Matrix{T}, A::AbstractArray{T,3}) where {T}
	@tensor gnew[-1, -2] := conj(A[1, 3, -1]) * g[1, 2] * A[2, 3, -2]
	return gnew
end

function _g_transfer_right(g::Matrix{T}, A::AbstractArray{T,3}) where {T}
	@tensor gnew[-1, -2] := conj(A[-1, 3, 1]) * g[1, 2] * A[-2, 3, 2]
	return gnew
end

# ---------- local normal equation at one site (matrix-free, KrylovKit) ----------

# model amplitude of sample k with the site tensor w: gₖ = ℓₖ·w·(e(xₖ)⊗rₖ)
# (the plain contraction — the conjugations of the normal equation live in the
# Hessian/right-hand side below, never here)
@inline function _model_amplitude(Lm::AbstractMatrix{T}, Rm::AbstractMatrix{T},
								  w::AbstractArray{T,3}, k::Integer, p::Integer) where {T}
	Dl, d, Dr = size(w)
	return sum(reshape(@view(Lm[k, :]), Dl, 1) .* (@view w[:, p, :]) .*
			   reshape(@view(Rm[k, :]), 1, Dr))
end

# the local data Hessian action: (M·z)[a,p,b] = Σₖ conj(ℓₖ[a]rₖ[b])·gₖ(z)·δ_{p,xₖ}
# with gₖ = fₖᵀ·z; plus the ridge α·(g_s ⊗ I_d ⊗ g_{s+1})·z (seq2seq's HS regularizer
# in the current gauge — Hermitian PSD, so CG applies)
function _ls_apply(c::ALSReconCache, s::Integer, z::AbstractArray{T,3}, α::Real) where {T}
	Dl, d, Dr = size(z)
	out = zeros(T, Dl, d, Dr)
	Lm, Rm = c.Lmat[s], c.Rmat[s]
	for k in eachindex(c.a)
		p = c.X[s, k]
		gk = _model_amplitude(Lm, Rm, z, k, p)
		@views out[:, p, :] .+= gk .* conj(reshape(Lm[k, :], Dl, 1) .*
										   reshape(Rm[k, :], 1, Dr))
	end
	if α != 0
		gL, gR = c.gstorage[s], c.gstorage[s+1]
		for p in 1:d
			@views out[:, p, :] .+= α .* (gL * z[:, p, :] * transpose(gR))
		end
	end
	return out
end

# the local normal-equation right-hand side: b[a, p, b'] = Σₖ:xₖ=p aₖ·conj(ℓₖ[a]rₖ[b'])
# (the ∂ℒ/∂z̄ equation of the plain model amplitude ⟨𝐱ₖ|ψ⟩ = fₖᵀ·vec(ψ_s); the LS
# conjugations land exactly as written — a swapped one solves the conjugated state)
function _ls_rhs(c::ALSReconCache, s::Integer)
	Dl, d, Dr = size(c.ψ[s])
	b = zeros(eltype(c.a), Dl, d, Dr)
	Lm, Rm = c.Lmat[s], c.Rmat[s]
	for k in eachindex(c.a)
		@views b[:, c.X[s, k], :] .+= c.a[k] .* conj(reshape(Lm[k, :], Dl, 1) .*
													 reshape(Rm[k, :], 1, Dr))
	end
	return b
end

"""
	_ls_solve(c, s, alg) -> w

Solve the local normal equation (M + α·R)·w = b iteratively with KrylovKit's
`linsolve` (the linear operator is applied matrix-free through the per-sample
features; the current site tensor is the warm start). Returns the updated site
tensor `w` of shape (Dl, d, Dr).
"""
function _ls_solve(c::ALSReconCache, s::Integer, alg::ALSRecon)
	b = _ls_rhs(c, s)
	w, _ = KrylovKit.linsolve(z -> _ls_apply(c, s, z, alg.α), b, c.ψ[s];
							  ishermitian=true, tol=Defaults.tol, krylovdim=25, maxiter=100)
	return w
end

# exact global data objective after the site update (ridge excluded), from the
# per-sample model amplitudes of the updated tensor: ℒ = Σₖ |gₖ − aₖ|²
function _site_loss(c::ALSReconCache, s::Integer, w::AbstractArray)
	Lm, Rm = c.Lmat[s], c.Rmat[s]
	loss = 0.0
	for k in eachindex(c.a)
		gk = _model_amplitude(Lm, Rm, w, k, c.X[s, k])
		loss += abs2(gk - c.a[k])
	end
	return loss
end

# ---------- ALS sweeps ----------

"""
	leftsweep!(c::ALSReconCache, alg::ALSRecon) -> kvals

Left-to-right single-site ALS sweep: matrix-free KrylovKit local normal-equation
solves, QR gauge moves, left-context and ridge transfers. `kvals[s]` is the exact
global data objective after updating site `s` (non-increasing; the ridge is excluded).
"""
function leftsweep!(c::ALSReconCache, alg::ALSRecon)
	L = length(c.ψ)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		w = _ls_solve(c, s, alg)
		kvals[s] = _site_loss(c, s, w)
		q, r = _gauge_left(w)
		c.ψ[s] = q
		c.ψ[s+1] = _contract_first(c.ψ[s+1], r)
		_left_transfer!(c, s)
		_g_transfer_left!(c, s)
	end
	w = _ls_solve(c, L, alg)
	kvals[L] = _site_loss(c, L, w)
	c.ψ[L] = w
	return kvals
end

"""
	rightsweep!(c::ALSReconCache, alg::ALSRecon) -> kvals

Right-to-left ALS sweep (symmetric, LQ gauge moves). `kvals` is ordered by processing
time (sites `L, …, 1`); the losses are non-increasing.
"""
function rightsweep!(c::ALSReconCache, alg::ALSRecon)
	L = length(c.ψ)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		w = _ls_solve(c, s, alg)
		kvals[k] = _site_loss(c, s, w)
		k += 1
		l, q = _gauge_right(w)
		c.ψ[s] = q
		c.ψ[s-1] = _contract_last(c.ψ[s-1], l)
		_right_transfer!(c, s)
		_g_transfer_right!(c, s)
	end
	w = _ls_solve(c, 1, alg)
	kvals[L] = _site_loss(c, 1, w)
	c.ψ[1] = w
	return kvals
end

sweep!(c::ALSReconCache, alg::ALSRecon) = vcat(leftsweep!(c, alg), rightsweep!(c, alg))

# ---------- global data objective & amplitudes ----------

# amplitudes ⟨𝐱⁽ᵏ⁾|ψ⟩ of all samples, by per-sample chain folding
function _amplitudes(c::ALSReconCache)
	v = ones(eltype(c.a), size(c.X, 2), 1)
	for s in 1:length(c.ψ)
		v = _left_context_transfer(v, c.ψ[s], @view(c.X[s, :]))
	end
	return vec(v)
end

_ls_loss(c::ALSReconCache) = sum(abs2, _amplitudes(c) .- c.a)

# amplitude ⟨𝐱|ψ⟩ at one coordinate (adaptive-residual evaluation)
function _amplitude(ψ::CanonicalMPS, 𝐱::NTuple{L,Int}) where {L}
	L == 1 && return ψ[1][1, 𝐱[1], 1]
	v = reshape(ψ[1][1, 𝐱[1], :], 1, :)
	for s in 2:L-1
		v = v * ψ[s][:, 𝐱[s], :]
	end
	return (v * ψ[L][:, 𝐱[L], :])[1]
end

_random_coords(ds::NTuple{L,Int}, n::Integer) where {L} =
	Vector{NTuple{L,Int}}([Tuple(rand(1:ds[j]) for j in 1:L) for _ in 1:n])

# ---------- drivers ----------

"""
	reconstruct(samples, ds, alg::ALSRecon = ALSRecon()) -> (ψ, traj)

Fit an MPS `ψ` to a fixed sample set of tensor amplitudes by single-site ALS sweeps
(`samples::Vector` of `Pair{NTuple{L,Int},T}` or `(𝐱, a)` tuples). The initial guess
is a random chain of bond `alg.D`. Returns `ψ` (scaling 1; the represented amplitudes
are fitted) and `traj`, the per-sweep loss history of `iterative_compute!`.
"""
function reconstruct(samples, ds::NTuple{L,Int}, alg::ALSRecon = ALSRecon()) where {L}
	ψ = randommps(ComplexF64, collect(ds); D=alg.D, normalize=false)
	ψ, traj = reconstruct!(ψ, samples, alg)
	return ψ, traj
end

"""
	reconstruct!(ψ, samples, alg::ALSRecon = ALSRecon()) -> (ψ, traj)

In-place variant of [`reconstruct`](@ref): refine the provided chain on the fixed
sample set (the per-site scaling is folded into the data and the bond profile is
re-fitted to `alg.D` with `changebond!`). Returns `(ψ, traj)`.
"""
function reconstruct!(ψ::CanonicalMPS, samples, alg::ALSRecon = ALSRecon())
	ds = Tuple(phydims(ψ))
	X, a = _normalize_samples(ds, samples)
	bonddim(ψ) != alg.D && changebond!(ψ; D=alg.D)
	c = ALSReconCache(ψ, X, a)
	traj = iterative_compute!(c, alg)
	return ψ, traj
end

"""
	reconstruct(Afun, ds, alg::ALSRecon = ALSRecon())
	-> (ψ, info::NamedTuple{(:loss, :maxres, :nsamples, :rounds)})

Adaptive reconstruction from a black-box amplitude oracle `Afun(𝐱)` (a dense tensor
`A` is wrapped by the caller as `Afun = (𝐱) -> A[𝐱...]`): alternate the inner ALS fit
on the current sample set with residual-driven sample enrichment — each round evaluates
`|⟨𝐱′|ψ⟩ − A(𝐱′)|` on a fresh random candidate pool (`alg.nbuffer` points) and adds the
`alg.nadd` worst points to the sample set (the least-squares analog of TCI's pivot
selection). Stops when the maximal pool residual drops below `alg.tol` or after
`alg.maxiter` rounds.
"""
function reconstruct(Afun::Function, ds::NTuple{L,Int}, alg::ALSRecon = ALSRecon()) where {L}
	ψ = randommps(ComplexF64, collect(ds); D=alg.D, normalize=false)
	S = [(𝐱, Afun(𝐱)) for 𝐱 in _random_coords(ds, alg.nbuffer)]
	maxres = Inf
	rounds = 0
	while rounds < alg.maxiter
		reconstruct!(ψ, S, alg)
		rounds += 1
		cand = _random_coords(ds, alg.nbuffer)
		ρ = [abs(Afun(𝐱) - _amplitude(ψ, 𝐱)) for 𝐱 in cand]
		maxres = maximum(ρ)
		(alg.verbosity > 0) &&
			println("ALSRecon round $rounds: maxres = $(round(maxres; sigdigits=4))")
		(maxres < alg.tol || rounds >= alg.maxiter) && break
		worst = partialsortperm(ρ, 1:min(alg.nadd, length(ρ)); rev=true)
		append!(S, [(𝐱, Afun(𝐱)) for 𝐱 in cand[worst]])
	end
	X, a = _normalize_samples(ds, S)
	c = ALSReconCache(ψ, X, a)
	return ψ, (loss = _ls_loss(c), maxres = maxres, nsamples = length(S), rounds = rounds)
end
