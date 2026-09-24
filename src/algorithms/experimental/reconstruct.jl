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
	ALSRecon(; maxiter=Defaults.maxiter, tol=Defaults.tol, D=Defaults.D, α=1.0e-4,
	         solver=DefaultLinearSolver, nbuffer=1024, nadd=128, nrounds=20, verbosity=0)

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
part of the reported loss). The local normal equations are solved by the KrylovKit
iterative solver `alg.solver`. `nbuffer`/`nadd`/`nrounds` drive the adaptive
sample-enrichment loop of [`reconstruct`](@ref): an initial random pool of `nbuffer`
oracle queries, then `nadd` Born samples of the current fit (re-evaluated by the
oracle) per round, for at most `nrounds` rounds.
"""
@kwdef struct ALSRecon{S<:KrylovKit.LinearSolver} <: SingleSiteUpdate
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	D::Int = Defaults.D
	α::Float64 = 1.0e-4      # HS ridge on the local solves (seq2seq regularizer)
	solver::S = DefaultLinearSolver
	nbuffer::Int = 1024      # initial random sample pool of the adaptive loop
	nadd::Int = 128          # Born samples of the current fit added per round
	nrounds::Int = 20        # adaptive enrichment round cap
	verbosity::Int = 0
end

"""
	ALSRecon2(; maxiter=Defaults.maxiter, tol=Defaults.tol, trunc=truncdim(D=Defaults.D),
	          α=1.0e-4, solver=DefaultLinearSolver, nbuffer=1024, nadd=128, nrounds=20,
	          verbosity=0)

Two-site variant of [`ALSRecon`](@ref): the neighboring site pair is optimized jointly
(the same rank-4 local normal equation structure, solved matrix-free by `alg.solver`)
and re-split by a truncating SVD under `alg.trunc::TruncationScheme`, so the bond
dimension adapts during the sweeps — growth where the samples demand it, truncation
where the scheme caps it. The single-site ALS losses stay monotone; the two-site
re-split further relieves local minima of the single-site problem. Shares the
adaptive sample-enrichment loop of [`reconstruct`](@ref).
"""
@kwdef struct ALSRecon2{TR<:TruncationScheme,S<:KrylovKit.LinearSolver} <: TwoSiteUpdate
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	trunc::TR = truncdim(D=Defaults.D)
	α::Float64 = 1.0e-4      # HS ridge on the local solves (seq2seq regularizer)
	solver::S = DefaultLinearSolver
	nbuffer::Int = 1024      # initial random sample pool of the adaptive loop
	nadd::Int = 128          # Born samples of the current fit added per round
	nrounds::Int = 20        # adaptive enrichment round cap
	verbosity::Int = 0
end

# ---------- sample bookkeeping ----------

# samples: Vector of Pair{Vector{Int},T} or (𝐱, a) tuples with 𝐱::Vector{Int}
# → (X::Vector{Vector{Int}} (the coordinates), a::Vector (the amplitudes))
function _normalize_samples(ds::Vector{Int}, samples)
	N = length(samples)
	N > 0 || throw(ArgumentError("empty sample set"))
	X = Vector{Vector{Int}}(undef, N)
	vals = [smp isa Pair ? last(smp) : smp[2] for smp in samples]
	T = promote_type(eltype(vals), Float64)
	a = Vector{T}(undef, N)
	for (k, smp) in enumerate(samples)
		𝐱 = smp isa Pair ? first(smp) : smp[1]
		length(𝐱) == length(ds) ||
			throw(DimensionMismatch("sample coordinates must have length $(length(ds))"))
		for (j, xj) in enumerate(𝐱)
			1 <= xj <= ds[j] ||
				throw(ArgumentError("sample coordinate out of range: site $j value $xj"))
		end
		X[k] = convert(Vector{Int}, 𝐱)
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
	X::Vector{Vector{Int}}
	a::Vector{T}
	Lmat::Vector{Matrix{T}}
	Rmat::Vector{Matrix{T}}
	gstorage::Vector{Matrix{T}}
end

function ALSReconCache(ψ::CanonicalMPS, X::Vector{Vector{Int}}, a::Vector)
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

# the site-`s` physical value of every sample: [X[1][s], …, X[N][s]]
_xcol(c::ALSReconCache, s::Integer) = [x[s] for x in c.X]

# right-to-left initialization of the right contexts and the ridge stack: Rmat[s]
# covers sites s+1..L, gstorage[s] is the HS overlap over sites s..L; Lmat[1] is the
# trivial boundary (the left contexts are transferred during the sweeps)
function _init_contexts_right!(c::ALSReconCache)
	L = length(c.ψ)
	for s in L:-1:1
		s < L && (c.Rmat[s] = _right_context_transfer(c.Rmat[s+1], c.ψ[s+1], _xcol(c, s + 1)))
		c.gstorage[s] = _g_transfer_right(c.gstorage[s+1], c.ψ[s])
	end
	return c
end

# sweep-time transfers: ONE stack entry per site (the next site's left context /
# left ridge); the right contexts of the not-yet-reached sites stay untouched
# (the dmrg2.jl double-update lesson)
_left_transfer!(c::ALSReconCache, s) =
	(c.Lmat[s+1] = _left_context_transfer(c.Lmat[s], c.ψ[s], _xcol(c, s)); c)
_right_transfer!(c::ALSReconCache, s) =
	(c.Rmat[s-1] = _right_context_transfer(c.Rmat[s], c.ψ[s], _xcol(c, s)); c)
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
	xs = _xcol(c, s)
	for k in eachindex(c.a)
		p = xs[k]
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
	Lm, Rm, xs = c.Lmat[s], c.Rmat[s], _xcol(c, s)
	for k in eachindex(c.a)
		@views b[:, xs[k], :] .+= c.a[k] .* conj(reshape(Lm[k, :], Dl, 1) .*
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
	w, _ = KrylovKit.linsolve(z -> _ls_apply(c, s, z, alg.α), b, c.ψ[s], alg.solver)
	return w
end

# exact global data objective after the site update (ridge excluded), from the
# per-sample model amplitudes of the updated tensor: ℒ = Σₖ |gₖ − aₖ|²
function _site_loss(c::ALSReconCache, s::Integer, w::AbstractArray)
	Lm, Rm, xs = c.Lmat[s], c.Rmat[s], _xcol(c, s)
	loss = 0.0
	for k in eachindex(c.a)
		gk = _model_amplitude(Lm, Rm, w, k, xs[k])
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

# ---------- two-site (ALSRecon2) sweeps ----------

# model amplitude of sample k with the pair tensor w: gₖ = ℓₖ·w[:,xₛ,xₛ₊₁,:]·rₖ
@inline function _model2_amplitude(Lm::AbstractMatrix{T}, Rm::AbstractMatrix{T},
								   w::AbstractArray{T,4}, k::Integer, p::Integer,
								   q::Integer) where {T}
	Dl, Dr = size(w, 1), size(w, 4)
	return sum(reshape(@view(Lm[k, :]), Dl, 1) .* (@view w[:, p, q, :]) .*
			   reshape(@view(Rm[k, :]), 1, Dr))
end

# the two-site local data Hessian action: (M·z)[a,p,q,b] = Σₖ:gₖ = ℓₖᵀz e ⊗ rₖ
# conj(ℓₖ[a]rₖ[b])·δ_{p,xₛ}δ_{q,xₛ₊₁}; plus the ridge α·(g_s ⊗ I ⊗ I ⊗ g_{s+2})·z
function _ls2_apply(c::ALSReconCache, s::Integer, z::AbstractArray{T,4}, α::Real) where {T}
	Dl, d1, d2, Dr = size(z)
	out = zeros(T, Dl, d1, d2, Dr)
	Lm, Rm = c.Lmat[s], c.Rmat[s+1]
	xs, xs2 = _xcol(c, s), _xcol(c, s + 1)
	for k in eachindex(c.a)
		p, q = xs[k], xs2[k]
		gk = _model2_amplitude(Lm, Rm, z, k, p, q)
		@views out[:, p, q, :] .+= gk .* conj(reshape(Lm[k, :], Dl, 1) .*
											  reshape(Rm[k, :], 1, Dr))
	end
	if α != 0
		gL, gR = c.gstorage[s], c.gstorage[s+2]
		for q in 1:d2, p in 1:d1
			@views out[:, p, q, :] .+= α .* (gL * z[:, p, q, :] * transpose(gR))
		end
	end
	return out
end

# the two-site normal-equation right-hand side:
# b[a,p,q,b'] = Σₖ:(xₛ,xₛ₊₁)=(p,q) aₖ·conj(ℓₖ[a]rₖ[b'])
function _ls2_rhs(c::ALSReconCache, s::Integer)
	Dl, d1, d2, Dr = size(c.ψ[s], 1), size(c.ψ[s], 2), size(c.ψ[s+1], 2), size(c.ψ[s+1], 3)
	b = zeros(eltype(c.a), Dl, d1, d2, Dr)
	Lm, Rm, xs, xs2 = c.Lmat[s], c.Rmat[s+1], _xcol(c, s), _xcol(c, s + 1)
	for k in eachindex(c.a)
		@views b[:, xs[k], xs2[k], :] .+= c.a[k] .* conj(reshape(Lm[k, :], Dl, 1) .*
														 reshape(Rm[k, :], 1, Dr))
	end
	return b
end

"""
	_ls2_solve(c, s, alg) -> w

Solve the two-site local normal equation (M + α·R)·w = b iteratively with KrylovKit's
`linsolve` (matrix-free through the per-sample features; the pair tensor of the current
chain is the warm start). Returns the updated pair tensor of shape (Dl, d, d, Dr).
"""
function _ls2_solve(c::ALSReconCache, s::Integer, alg::ALSRecon2)
	b = _ls2_rhs(c, s)
	w0 = @tensor w0[a, p, q, b] := c.ψ[s][a, p, j] * c.ψ[s+1][j, q, b]
	w, _ = KrylovKit.linsolve(z -> _ls2_apply(c, s, z, alg.α), b, w0, alg.solver)
	return w
end

# exact global data objective after the pair update (ridge excluded)
function _ls2_loss(c::ALSReconCache, s::Integer, w::AbstractArray)
	Lm, Rm, xs, xs2 = c.Lmat[s], c.Rmat[s+1], _xcol(c, s), _xcol(c, s + 1)
	loss = 0.0
	for k in eachindex(c.a)
		gk = _model2_amplitude(Lm, Rm, w, k, xs[k], xs2[k])
		loss += abs2(gk - c.a[k])
	end
	return loss
end

function leftsweep!(c::ALSReconCache, alg::ALSRecon2)
	L = length(c.ψ)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		w = _ls2_solve(c, s, alg)
		kvals[s] = _ls2_loss(c, s, w)
		_als2_update!(c.ψ, s, w, alg; move_right=true)
		_left_transfer!(c, s)
		_g_transfer_left!(c, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

function rightsweep!(c::ALSReconCache, alg::ALSRecon2)
	L = length(c.ψ)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		w = _ls2_solve(c, s - 1, alg)
		kvals[k] = _ls2_loss(c, s - 1, w)
		k += 1
		_als2_update!(c.ψ, s - 1, w, alg; move_right=false)
		_right_transfer!(c, s)
		_g_transfer_right!(c, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(c::ALSReconCache, alg::ALSRecon2) = vcat(leftsweep!(c, alg), rightsweep!(c, alg))

# ---------- global data objective & amplitudes ----------

_random_coords(ds::Vector{Int}, n::Integer) =
	[[rand(1:ds[j]) for j in eachindex(ds)] for _ in 1:n]

# ---------- drivers ----------

# the bond cap of the automatically drawn initial guesses
_initial_bond(alg::ALSRecon) = alg.D
_initial_bond(alg::ALSRecon2) = _guess_bond(alg.trunc)

"""
	reconstruct(samples, ds, alg = ALSRecon()) -> (ψ, traj)

Fit an MPS `ψ` to a fixed sample set of tensor amplitudes by ALS sweeps
(`samples::Vector` of `Pair{Vector{Int},T}` or `(𝐱, a)` tuples with
`𝐱::Vector{Int}`; `alg` is an [`ALSRecon`](@ref) or [`ALSRecon2`](@ref)). The
initial guess is a random chain of the bond profile demanded by `alg`. Returns `ψ`
(scaling 1; the represented amplitudes are fitted) and `traj`, the per-sweep loss
history of `iterative_compute!`.
"""
function reconstruct(samples, ds::Vector{Int}, alg::Union{ALSRecon,ALSRecon2} = ALSRecon())
	# normalized init: an unnormalized random chain of bond D carries amplitudes of
	# magnitude ~(d·D²)^(L/2), which wrecks the conditioning of the local normal
	# equations (the CG solves make no progress at all)
	ψ = randommps(ComplexF64, ds; D=_initial_bond(alg), normalize=true)
	ψ, traj = reconstruct!(ψ, samples, alg)
	return ψ, traj
end

"""
	reconstruct!(ψ, samples, alg = ALSRecon()) -> (ψ, traj)

In-place variant of [`reconstruct`](@ref): refine the provided chain on the fixed
sample set (the per-site scaling is folded into the data and the bond profile is
re-fitted to `alg`'s demand with `changebond!`). Returns `(ψ, traj)`.
"""
function reconstruct!(ψ::CanonicalMPS, samples, alg::Union{ALSRecon,ALSRecon2} = ALSRecon())
	ds = phydims(ψ)
	X, a = _normalize_samples(ds, samples)
	D0 = _initial_bond(alg)
	bonddim(ψ) != D0 && changebond!(ψ; D=D0)
	c = ALSReconCache(ψ, X, a)
	traj = iterative_compute!(c, alg)
	return ψ, traj
end

"""
	reconstruct(Afun, ds, alg = ALSRecon())
	-> (ψ, info::NamedTuple{(:loss, :nsamples, :rounds)})

Adaptive black-box reconstruction: alternate the inner ALS fit on the current sample
pool with **active sample enrichment** — each round queries the oracle on `alg.nadd`
fresh configurations, half drawn from the Born distribution of the current fit
(`sample`, exploitation) and half uniformly random (exploration: an overfitted chain
concentrates its Born distribution on the training points, so pure Born sampling
would starve the loop of new information). This is the least-squares analog of TCI's
pivot selection. The initial pool is `alg.nbuffer` uniformly random queries.

The fresh Born samples act as the round's **held-out validation set** before joining
the training pool: the loop stops when the validation loss drops below `alg.tol` or
stagnates (relative change < `alg.tol`), or after `alg.nrounds` rounds. The
validation-based criterion is essential — an overparameterized chain can drive the
*training* loss to zero on any sample set, which carries no generalization
information. Each round runs the inner sweeps of `iterative_compute!` up to
`alg.maxiter`.
"""
function reconstruct(Afun::Function, ds::Vector{Int},
					 alg::Union{ALSRecon,ALSRecon2} = ALSRecon())
	ψ = randommps(ComplexF64, ds; D=_initial_bond(alg), normalize=true)
	seen = Set{Vector{Int}}()
	x0 = _random_coords(ds, 1)[1]
	S = Tuple{Vector{Int},typeof(Afun(x0))}[(x0, Afun(x0))]
	push!(seen, x0)
	for x in _random_coords(ds, alg.nbuffer - 1)
		(x in seen) && continue
		push!(seen, x)
		push!(S, (x, Afun(x)))
	end
	prevloss = Inf
	loss = Inf
	rounds = 0
	while rounds < alg.nrounds
		reconstruct!(ψ, S, alg)
		rounds += 1
		# active enrichment: half Born samples of the current fit (exploit — queries
		# concentrate where the fit puts weight), half uniform random (explore — an
		# overfitted chain concentrates its Born distribution on the training points,
		# so pure Born sampling would starve the loop of fresh information). The
		# sweeps leave the chain gauged arbitrarily; re-canonicalize (no truncation)
		# so that `sample` follows the exact Born distribution |ψ(𝐱)|². The fresh
		# points validate the fit BEFORE joining the training pool.
		unset_svectors!(ψ)
		cand = vcat(sample(ψ, alg.nadd ÷ 2),
					_random_coords(ds, alg.nadd - alg.nadd ÷ 2))
		xs = [x for x in cand if x ∉ seen]
		vals = [Afun(x) for x in xs]
		loss = isempty(vals) ? 0.0 :
			   sum(abs2(amplitude(ψ, x) - v) for (x, v) in zip(xs, vals))
		union!(seen, xs)
		append!(S, zip(xs, vals))
		(alg.verbosity > 0) &&
			println("reconstruct round $rounds: nsamples = $(length(S)), val loss = ",
					round(loss; sigdigits=4))
		delta = (isfinite(prevloss) && !iszero(prevloss)) ?
				abs(loss - prevloss) / prevloss : Inf
		(loss < alg.tol || delta < alg.tol) && break
		prevloss = loss
	end
	return ψ, (loss = loss, nsamples = length(S), rounds = rounds)
end
