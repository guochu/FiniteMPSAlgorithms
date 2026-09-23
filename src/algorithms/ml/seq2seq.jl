# seq2seq: variational (DMRG-style ALS) fitting of an MPO to a dataset of MPS pairs,
# following guochu/MPSLearning.jl's `OptimizeMPO`. Given training pairs {(x_n, y_n)},
# find a finite-bond MPO W (input dims = dims of x, output dims = dims of y) minimizing
#   F(W) = Σ_n ||W·x_n − y_n||² + alpha·||W||²_HS
# by single-site ALS sweeps. Three environment stacks are swept simultaneously:
#   hstorage[n]: ⟨x_n|W†(·)W|x_n⟩  quadratic stack (axes: xL_bra, W-bra, W-ket, xL_ket)
#   bstorage[n]: ⟨x_n|W†(·)|y_n⟩   linear stack    (axes: xL, W-bond, yL)
#   gstorage:    ⟨W|(·)⟩_HS        ridge stack    (axes: W-bra, W-ket)
# At each site the local normal equation (Σ_n H_n + alpha·R)·w = Σ_n t_n is solved
# iteratively with KrylovKit's `linsolve` (matrix-free on the operator action; the
# current site tensor is the warm start); the gauge is moved by QR/LQ and the stacks
# are incremented. Every per-site loss is the exact global data objective
# Σ_n||W·x_n − y_n||²/N at that point of the sweep (the ridge regularizes the solve
# but is not part of the reported loss).

"""
	Seq2Seq(; maxiter=Defaults.maxiter, tol=Defaults.tol, D=Defaults.D, α=1.0e-4,
	        nadd=8, nbuffer=1024, verbosity=0)

Algorithm configuration of the `seq2seq` MPO fit: single-site ALS sweeps with the
bond profile `alg.D` and the Hilbert-Schmidt ridge `α·‖W‖²_HS` added to the local
normal equations for conditioning (the seq2seq regularization; not part of the
reported loss). The local normal equations are solved densely (a direct `\` on the
explicit local Hessian). `nadd`/`nbuffer` drive the adaptive data-enrichment loop of the
oracle-based entry point `seq2seq(pairfun, dxs, dys, alg)` (unused by the
fixed-dataset entry points).
"""
@kwdef struct Seq2Seq <: SingleSiteUpdate
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	D::Int = Defaults.D
	α::Float64 = 1.0e-4      # HS ridge on the local solves
	nadd::Int = 8            # pairs added per adaptive enrichment round
	nbuffer::Int = 1024      # random candidate pool of the residual evaluation
	verbosity::Int = 0
end

# ---------- environment transfer primitives ----------
# Site tensors: W[aL, po, aR, pi] (po = output/y side, pi = input/x side), x/y[aL, p, aR].

"""
	_h_updateleft(::Array{<:Any,4}, W, x) -> h′

⟨x|W†W|x⟩ left transfer; `h` axes (xL_bra, W-bra, W-ket, xL_ket). The physical deltas
(W†·po ↔ W·po, W†·pi ↔ conj(x)·p, W·pi ↔ x·p) are contracted inside.
"""
function _h_updateleft(hold::AbstractArray{T,4}, W::MPOTensor, x::MPSTensor) where {T}
	@tensor hnew[-1, -2, -3, -4] :=
		conj(x[1, 5, -1]) * conj(W[2, 8, -2, 5]) * hold[1, 2, 3, 4] * W[3, 8, -3, 7] * x[4, 7, -4]
	return hnew
end

function _h_updateright(hold::AbstractArray{T,4}, W::MPOTensor, x::MPSTensor) where {T}
	@tensor hnew[-1, -2, -3, -4] :=
		conj(x[-1, 6, 1]) * conj(W[-2, 5, 2, 6]) * hold[1, 2, 3, 4] * W[-3, 5, 3, 7] * x[-4, 7, 4]
	return hnew
end

# the ⟨x|W†|y⟩ linear transfers (`_b_updateleft` / `_b_updateright`) are shared with linsolve.jl

"""
	_g_updateleft(::Array{<:Any,2}, W) -> g′

Hilbert-Schmidt ridge ⟨W|W⟩ left transfer; `g` axes (W-bra, W-ket).
"""
function _g_updateleft(hold::AbstractArray{T,2}, W::MPOTensor) where {T}
	@tensor gnew[-1, -2] := conj(W[1, 3, -1, 4]) * hold[1, 2] * W[2, 3, -2, 4]
	return gnew
end

function _g_updateright(hold::AbstractArray{T,2}, W::MPOTensor) where {T}
	@tensor gnew[-1, -2] := conj(W[-1, 3, 1, 4]) * hold[1, 2] * W[-2, 3, 2, 4]
	return gnew
end

# ---------- ALS cache ----------

"""
Seq2SeqCache: ALS problem carrier for [`seq2seq`](@ref) — fit W minimizing
Σ_n ||W·x_n − y_n||² (+ alpha·||W||²_HS ridge on the local solves) with single-site
sweeps over the three (quadratic / linear / ridge) environment stacks.
"""
struct Seq2SeqCache{O, X, Y, T}
	H::O                                 # the variational MPO (mutated in place)
	kets::Vector{X}                      # folded inputs  (scaling = 1)
	bras::Vector{Y}                      # folded targets (scaling = 1)
	hstorage::Vector{Vector{Array{T,4}}}
	bstorage::Vector{Vector{Array{T,3}}}
	gstorage::Vector{Array{T,2}}
	ynorm::Float64
end

function _updateleft!(m::Seq2SeqCache, s)
	W = m.H[s]
	for n in eachindex(m.kets)
		m.hstorage[n][s+1] = _h_updateleft(m.hstorage[n][s], W, m.kets[n][s])
		m.bstorage[n][s+1] = _b_updateleft(m.bstorage[n][s], m.kets[n][s], W, m.bras[n][s])
	end
	m.gstorage[s+1] = _g_updateleft(m.gstorage[s], W)
	return m
end

function _updateright!(m::Seq2SeqCache, s)
	W = m.H[s]
	for n in eachindex(m.kets)
		m.hstorage[n][s] = _h_updateright(m.hstorage[n][s+1], W, m.kets[n][s])
		m.bstorage[n][s] = _b_updateright(m.bstorage[n][s+1], m.kets[n][s], W, m.bras[n][s])
	end
	m.gstorage[s] = _g_updateright(m.gstorage[s+1], W)
	return m
end

function _init_storages_right!(m::Seq2SeqCache)
	L = length(m.H)
	T = scalartype(m.H)
	m.gstorage[1] = ones(T, 1, 1)
	m.gstorage[L+1] = ones(T, 1, 1)
	for n in eachindex(m.kets)
		m.hstorage[n][1] = ones(T, 1, 1, 1, 1)
		m.hstorage[n][L+1] = ones(T, 1, 1, 1, 1)
		m.bstorage[n][1] = ones(T, 1, 1, 1)
		m.bstorage[n][L+1] = ones(T, 1, 1, 1)
	end
	for s in L:-1:2
		for n in eachindex(m.kets)
			m.hstorage[n][s] = _h_updateright(m.hstorage[n][s+1], m.H[s], m.kets[n][s])
			m.bstorage[n][s] = _b_updateright(m.bstorage[n][s+1], m.kets[n][s], m.H[s], m.bras[n][s])
		end
		m.gstorage[s] = _g_updateright(m.gstorage[s+1], m.H[s])
	end
	return m
end

# ---------- local normal equation at one site (dense solve) ----------

# the per-sample local Hessian action: (Hₙ·z)[aLb, po, aRb, piB] — the xₙ bra/ket
# physicals wrap the W†(·)W transfers carried by the hL/hR environments
function _h_apply(z::AbstractArray{T,4}, x::MPSTensor,
				  hL::AbstractArray{T,4}, hR::AbstractArray{T,4}) where {T}
	@tensor y[aLb, po, aRb, piB] := conj(x[xLb, piB, xRb]) *
		hL[xLb, aLb, aLk, xLk] * hR[xRb, aRb, aRk, xRk] *
		z[aLk, po, aRk, piK] * x[xLk, piK, xRk]
	return y
end

# Σₙ Hₙ·z — the data part of the normal-equation action
function _h_apply_all(m::Seq2SeqCache, s::Integer, z)
	y = zeros(eltype(z), size(z))
	for n in eachindex(m.kets)
		y .+= _h_apply(z, m.kets[n][s], m.hstorage[n][s], m.hstorage[n][s+1])
	end
	return y
end

# the ridge action: α·(g_s ⊗ I ⊗ g_{s+1})·z (seq2seq's HS regularizer in the current
# gauge — Hermitian PSD, so CG applies)
function _ridge_apply(m::Seq2SeqCache, s::Integer, z, α)
	lL, dy, lR, dx = size(z)
	out = zeros(eltype(z), size(z))
	gL, gR = m.gstorage[s], m.gstorage[s+1]
	for po in 1:dy, pi in 1:dx
		@views out[:, po, :, pi] .+= α .* (gL * z[:, po, :, pi] * transpose(gR))
	end
	return out
end

# t[aL, po, aR, pi]: local linear functional Σ_n ⟨x_n|W†|y_n⟩ with W's leg order
function _b_target_sum(m::Seq2SeqCache, s)
	W = m.H[s]
	t = zeros(scalartype(W), size(W))
	for n in eachindex(m.kets)
		t .+= _w_target(m.kets[n][s], m.bras[n][s], m.bstorage[n][s], m.bstorage[n][s+1])
	end
	return t
end

# linear target of one sample at site `s`: the coefficient of conj(W[aL, po, aR, pi]) in
# ⟨x|W†|y⟩, from the left env bL (xL_bra, aL_bra, yL) and right env bR (xR_bra, aR_bra, yR)
function _w_target(x::MPSTensor, y::MPSTensor,
				   bL::AbstractArray{T,3}, bR::AbstractArray{T,3}) where {T}
	@tensor t[-1, -2, -3, -4] :=
		bL[1, -1, 2] * conj(x[1, -4, 3]) * y[2, -2, 4] * bR[3, -3, 4]
	return t
end

function _site_solve(m::Seq2SeqCache, s::Integer, α::Real)
	W = m.H[s]
	lL, dy, lR, dx = size(W)
	T = scalartype(W)
	# the local Hessian decouples over the output physical `po`: a single
	# (aLb,aRb,piB)×(aLk,aRk,piK) matrix solves for every po at once — w, H and t stay
	# matrices (no dy² blow-up of the system)
	K = zeros(T, lL, lR, dx, lL, lR, dx)
	for n in eachindex(m.kets)
		x = m.kets[n][s]
		@tensor Kk[aLb, aRb, piB, aLk, aRk, piK] :=
			m.hstorage[n][s][xLb, aLb, aLk, xLk] * m.hstorage[n][s+1][xRb, aRb, aRk, xRk] *
			conj(x[xLb, piB, xRb]) * x[xLk, piK, xRk]
		K .+= Kk
	end
	δ = Matrix{T}(I, dx, dx)
	@tensor Kr[aLb, aRb, piB, aLk, aRk, piK] :=
		α * m.gstorage[s][aLb, aLk] * m.gstorage[s+1][aRb, aRk] * δ[piB, piK]
	H = reshape(K .+ Kr, lL * lR * dx, lL * lR * dx)
	t = _b_target_sum(m, s)
	# w = H \ t in matrix form: rows (aL, aR, pi), columns po
	Tm = reshape(permutedims(t, (1, 3, 4, 2)), lL * lR * dx, dy)
	X = H \ Tm
	w = reshape(permutedims(reshape(X, lL, lR, dx, dy), (1, 4, 2, 3)), lL, dy, lR, dx)
	return w, t
end

# the exact global data objective after the site update (ridge excluded), from the
# operator action and linear target: ℒ = z†(Σₙ Hₙ)z − 2·Re(z†t) + Σₙ‖yₙ‖²
function _site_loss(m::Seq2SeqCache, s::Integer, z::AbstractArray{T,4}, t::AbstractArray{T,4}) where {T}
	Hz = _h_apply_all(m, s, z)
	return real(dot(z, Hz)) - 2 * real(dot(z, t)) + m.ynorm
end

# ---------- ALS sweeps ----------

"""
	leftsweep!(m::Seq2SeqCache, alg) -> kvals

One left-to-right ALS sweep: at each site the local normal equation is solved densely,
the chain is moved by QR and all three environment stacks are incremented. `kvals`
collects the exact global data objective after every site update.
"""
function leftsweep!(m::Seq2SeqCache, alg::Seq2Seq)
	L = length(m.H)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		w, t = _site_solve(m, s, alg.α)
		kvals[s] = _site_loss(m, s, w, t)
		q, r = _gauge_left(w)
		m.H[s] = q
		m.H[s+1] = _contract_first(m.H[s+1], r)
		_updateleft!(m, s)
	end
	w, t = _site_solve(m, L, alg.α)
	kvals[L] = _site_loss(m, L, w, t)
	m.H[L] = w
	return kvals
end

"""
	rightsweep!(m::Seq2SeqCache, alg) -> kvals

One right-to-left ALS sweep (symmetric, LQ gauge moves). `kvals` is ordered by processing
time (sites `L, L-1, …, 1`); the losses are non-increasing.
"""
function rightsweep!(m::Seq2SeqCache, alg::Seq2Seq)
	L = length(m.H)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		w, t = _site_solve(m, s, alg.α)
		kvals[k] = _site_loss(m, s, w, t)
		k += 1
		l, q = _gauge_right(w)
		m.H[s] = q
		m.H[s-1] = _contract_last(m.H[s-1], l)
		_updateright!(m, s)
	end
	w, t = _site_solve(m, 1, alg.α)
	kvals[L] = _site_loss(m, 1, w, t)
	m.H[1] = w
	return kvals
end

sweep!(m::Seq2SeqCache, alg::IterativeMPSAlgorithm) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

# ---------- initial guess ----------

# random MPO with input dims `dxs` / output dims `dys` and the capped bond profile
function _random_seq2seq_mpo(::Type{T}, dxs::AbstractVector{Int},
							 dys::AbstractVector{Int}, D::Int) where {T<:Number}
	L = length(dxs)
	prof = max_bonddims([dxs[i] * dys[i] for i in 1:L], D)
	data = Vector{Array{T,4}}(undef, L)
	for i in 1:L
		dl = i == 1 ? 1 : prof[i]
		dr = i == L ? 1 : prof[i+1]
		# small entries: the ALS local solves converge globally from a near-zero guess,
		# while full-size random starts can trap the sweeps in local minima
		data[i] = 0.1 .* randn(T, dl, dys[i], dr, dxs[i])
	end
	return MPO(data)
end

# ---------- validation and driver ----------

function _validate_seq2seq(xs::Vector{<:CanonicalMPS}, ys::Vector{<:CanonicalMPS})
	(isempty(xs) || length(xs) != length(ys)) &&
		throw(DimensionMismatch("numbers of x and y must match and be nonzero"))
	L = length(xs[1])
	(L >= 2) || throw(ArgumentError("seq2seq requires at least 2 sites"))
	dxs = phydims(xs[1])
	dys = phydims(ys[1])
	for n in eachindex(xs)
		(length(xs[n]) == L && length(ys[n]) == L) ||
			throw(DimensionMismatch("all x and y must have the same length"))
		(phydims(xs[n]) == dxs) ||
			throw(DimensionMismatch("all x must have the same physical dimensions"))
		(phydims(ys[n]) == dys) ||
			throw(DimensionMismatch("all y must have the same physical dimensions"))
	end
	return dxs, dys
end

# fold a chain's per-site scaling into every site tensor: bounded per site (a scaling^L
# power is never materialized) and the represented chains are unchanged
function _fold_scaling_sites(ψ::CanonicalMPS)
	s = scaling(ψ)
	(==(s, 1)) && return ψ
	return CanonicalMPS([A * s for A in ψ.data]; scaling=one(s))
end

"""
	init_seq2seqcache(xs, ys, alg::Seq2Seq = Seq2Seq())

Build the `Seq2SeqCache` of the seq2seq fit: validate the dataset, fold the per-site
scalings of the inputs/targets into their site tensors and draw a random initial MPO
of bond dimension `alg.D` (the ridge strength is `alg.α`).
"""
function init_seq2seqcache(xs, ys, alg::Seq2Seq = Seq2Seq())
	dxs, dys = _validate_seq2seq(xs, ys)
	T = promote_type(scalartype(xs[1]), scalartype(ys[1]))
	ompo = _random_seq2seq_mpo(T, dxs, dys, alg.D)
	xsf = [_fold_scaling_sites(x) for x in xs]
	ysf = [_fold_scaling_sites(y) for y in ys]
	return Seq2SeqCache(ompo, xsf, ysf)
end

"""
	Seq2SeqCache(H, kets, bras)

Build the `Seq2SeqCache` of the seq2seq fit: allocate the per-sample h/b environment
stacks, the ridge stack and the target norm, and initialize everything. The ridge
strength α enters at solve time from the `Seq2Seq` algorithm.
"""
function Seq2SeqCache(H::AbstractMPO, kets, bras)
	T = scalartype(H)
	# H is optimized in place; the sweeps never touch Schmidt values (a plain MPO has
	# none at all), so reset them on the working guess
	H isa CanonicalMPO && unset_svectors!(H)
	m = Seq2SeqCache(H, kets, bras,
					 [Vector{Array{T,4}}(undef, length(H) + 1) for _ in eachindex(kets)],
					 [Vector{Array{T,3}}(undef, length(H) + 1) for _ in eachindex(kets)],
					 Vector{Array{T,2}}(undef, length(H) + 1),
					 sum(norm(bras[n])^2 for n in eachindex(bras)))
	_init_storages_right!(m)
	return m
end

"""
	seq2seq(xs, ys, alg::Seq2Seq = Seq2Seq()) -> (W, traj)

Fit an MPO `W` to a dataset of MPS pairs `(xs[n], ys[n])` by DMRG (single-site ALS)
sweeps, following guochu/MPSLearning.jl: minimize `Σ_n ||W·x_n − y_n||²`, with a
Hilbert-Schmidt ridge `α·||W||²` (`alg.α`) added to the local solves for conditioning
(the default `α = 0.01` follows MPSLearning; the ridge is not part of the
reported loss). The input (output) physical dimensions of `W` match the dimensions
of `xs` (`ys`), so rectangular maps `dx → dy` are supported. The initial guess is a
random MPO of bond dimension `alg.D`; the data scalings are folded into the site tensors
at entry. `traj` collects the exact global data objective after every site update,
grouped per sweep; each vector is in processing-time order (the left sweep sites
`1:L`, the right sweep sites `L:-1:1`), so the losses are non-increasing; convergence
follows the unified `iterative_compute!` criterion (relative difference of the last
loss of two successive sweeps below `alg.tol`).
"""
function seq2seq(xs::Vector{<:CanonicalMPS}, ys::Vector{<:CanonicalMPS},
				 alg::Seq2Seq = Seq2Seq())
	dxs, dys = _validate_seq2seq(xs, ys)
	ompo = _random_seq2seq_mpo(promote_type(scalartype(xs[1]), scalartype(ys[1])), dxs, dys, alg.D)
	traj = seq2seq!(ompo, xs, ys, alg)
	return ompo, traj
end

"""
	seq2seq!(W::AbstractMPO, xs, ys, alg::Seq2Seq = Seq2Seq()) -> traj

In-place variant of [`seq2seq`](@ref): fit the provided MPO `W` to the dataset
`(xs[n], ys[n])` (its site tensors are updated in place, so `W` doubles as the initial
guess). The data scalings are folded into the site tensors at entry. Returns `traj`,
the per-sweep loss history of `iterative_compute!`.
"""
function seq2seq!(W::AbstractMPO, xs::Vector{<:CanonicalMPS}, ys::Vector{<:CanonicalMPS},
				  alg::Seq2Seq = Seq2Seq())
	_validate_seq2seq(xs, ys)
	bonddim(W) != alg.D && changebond!(W; D=alg.D)
	xsf = [_fold_scaling_sites(x) for x in xs]
	ysf = [_fold_scaling_sites(y) for y in ys]
	m = Seq2SeqCache(W, xsf, ysf)
	return iterative_compute!(m, alg)
end

# ---------- adaptive oracle-based fit (the seq2seq analog of ALSRecon's enrichment) ----------

# random candidate inputs of the enrichment loop (the analog of ALSRecon's random
# coordinates): normalized random MPS of bond 2 over the input dimensions
_random_seq2seq_inputs(dxs::Vector{Int}, n::Integer) =
	[randommps(ComplexF64, dxs; D=2, normalize=true) for _ in 1:n]

function _seq2seq_pair(pairfun::Function, x::CanonicalMPS, dys::Vector{Int})
	y = pairfun(x)
	phydims(y) == dys ||
		throw(DimensionMismatch("oracle output dimensions $(phydims(y)) ≠ $dys"))
	return (x, y)
end

"""
	seq2seq(pairfun::Function, dxs, dys, alg::Seq2Seq = Seq2Seq())
		-> (W, info::NamedTuple{(:loss, :maxerr, :npairs, :rounds)})

Adaptive seq2seq fit from a black-box pair oracle `pairfun(x) -> y`: alternate the
inner ALS fit on the current training set with residual-driven data enrichment —
the seq2seq analog of ALSRecon's sample enrichment. Each round evaluates the
relative prediction error `‖W·x′ − y′‖/‖y′‖` of the current `W` on a fresh pool of
`alg.nbuffer` random input states `x′`, queries the oracle for the targets of the
`alg.nadd` worst inputs and adds those pairs to the training set. Stops when the
maximal pool error drops below `alg.tol` or after `alg.maxiter` rounds. Candidate
inputs are random normalized MPS of bond 2 over `dxs`. Returns `(W, info)` with the
final training loss, the maximal pool error, the number of training pairs and the
number of enrichment rounds.
"""
function seq2seq(pairfun::Function, dxs::Vector{Int}, dys::Vector{Int},
				 alg::Seq2Seq = Seq2Seq())
	length(dxs) == length(dys) ||
		throw(DimensionMismatch("dxs and dys must have equal length"))
	W = _random_seq2seq_mpo(ComplexF64, dxs, dys, alg.D)
	S = [_seq2seq_pair(pairfun, x, dys) for x in _random_seq2seq_inputs(dxs, alg.nbuffer)]
	traj = Vector{Vector{Float64}}()
	maxerr = Inf
	rounds = 0
	while rounds < alg.maxiter
		traj = seq2seq!(W, [p[1] for p in S], [p[2] for p in S], alg)
		rounds += 1
		cand = _random_seq2seq_inputs(dxs, alg.nbuffer)
		ys = [pairfun(x) for x in cand]
		errs = [distance(W * x, y) / max(norm(y), 1e-12) for (x, y) in zip(cand, ys)]
		maxerr = maximum(errs)
		(alg.verbosity > 0) &&
			println("Seq2Seq round $rounds: maxerr = $(round(maxerr; sigdigits=4))")
		(maxerr < alg.tol || rounds >= alg.maxiter) && break
		worst = partialsortperm(errs, 1:min(alg.nadd, length(errs)); rev=true)
		append!(S, [_seq2seq_pair(pairfun, cand[i], dys) for i in worst])
	end
	loss = isempty(traj) ? NaN : traj[end][end]
	return W, (loss = loss, maxerr = maxerr, npairs = length(S), rounds = rounds)
end
