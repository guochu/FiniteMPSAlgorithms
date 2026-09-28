# iterative (ALS) hadamard product: a finite-bond MPS `omps` approximating the
# pointwise product ψA ⊙ ψB. The template mirrors mult.jl and is dispatched on alg:
#   SVDCompression -> form the exact hadamard, then one SVD compression sweep
#   DMRG1          -> single-site variational ALS; one full sweep = leftsweep + rightsweep

# ---------- three-MPS environment transfers ----------
# hstorage[s]::Array{T,3} carries axes (omps-bond, A-bond, B-bond). The pointwise
# product shares ONE physical value, so the three physical indices contract together.

"""
	_updateleft(h, O, A, B) -> h′   (all chains MPSTensor)

Left-to-right transfer of the ⟨omps| ψA ⊙ ψB⟩ three-chain environment; the shared
physical index of all three site tensors is contracted.
"""
function _updateleft(hold::AbstractArray{T,3}, O::MPSTensor, A::MPSTensor, B::MPSTensor) where {T}
	# contraction order: fold hold into conj(O) first, then apply A and B batched over
	# the physical index (p is shared elementwise between A and B, NOT contracted).
	# Every intermediate stays at O(D^3) — materializing the fused pair KB would cost
	# O(D^4·d) memory and O(D^5·d) work per transfer.
	# conj(O) is hoisted: matmuls against conj-wrapped strided slices fall back to the
	# generic (non-BLAS) matmul, ~20x slower than the hoisted form (measured, D=32)
	dL, dp, dR = size(A, 1), size(A, 2), size(A, 3)
	dbL, dbR = size(B, 1), size(B, 3)
	dO = size(O, 3)
	hnew = zeros(T, dO, dR, dbR)
	Hm = reshape(permutedims(hold, (2, 3, 1)), dL * dbL, size(O, 1))   # (aL bL, oL)
	Oc = conj(O)
	for p in 1:dp
		M = Hm * @view(Oc[:, p, :])                         # (aL bL, oR)
		Y = reshape(transpose(@view(A[:, p, :])) * reshape(M, dL, dbL * dO), dR, dbL, dO)    # (aR, bL, oR)
		Y2 = reshape(permutedims(Y, (2, 1, 3)), dbL, dR * dO)         # (bL, aR oR)
		res = reshape(transpose(@view(B[:, p, :])) * Y2, dbR, dR, dO) # (bR, aR, oR)
		hnew .+= permutedims(res, (3, 2, 1))                         # (oR, aR, bR)
	end
	return hnew
end

"""
	_updateright(h, O, A, B) -> h′

Right-to-left transfer of the three-chain environment (symmetric to [`_updateleft`](@ref)).
"""
function _updateright(hold::AbstractArray{T,3}, O::MPSTensor, A::MPSTensor, B::MPSTensor) where {T}
	# mirror of `_updateleft`: fold hold into conj(O) first, then batch over p
	dL, dp, dR = size(A, 1), size(A, 2), size(A, 3)
	dbL, dbR = size(B, 1), size(B, 3)
	dO = size(O, 1)
	hnew = zeros(T, dO, dL, dbL)
	Hm = reshape(hold, size(O, 3), dR * dbR)               # (oR, aR bR)
	Oc = conj(O)   # hoisted: conj-wrapped slices fall back to generic (non-BLAS) matmul
	for p in 1:dp
		M = reshape(transpose(@view(Oc[:, p, :]) * Hm), dR, dbR, dO)  # (aR, bR, oL)
		Y = reshape(@view(A[:, p, :]) * reshape(M, dR, dbR * dO), dL, dbR, dO)  # (aL, bR, oL)
		Y2 = reshape(permutedims(Y, (2, 1, 3)), dbR, dL * dO)            # (bR, aL oL)
		res = reshape(@view(B[:, p, :]) * Y2, dbL, dL, dO)    # (bL, aL, oL)
		hnew .+= permutedims(res, (3, 2, 1))                            # (oL, aL, bL)
	end
	return hnew
end

# local ALS target: drive the omps site tensor towards the pointwise product
function _reduce_hadamard_site(A::MPSTensor, B::MPSTensor,
							   cleft::AbstractArray{T,3}, cright::AbstractArray{T,3}) where {T}
	# batched over the physical index (shared elementwise between A and B): fold B into
	# cright, then A, then cleft — all intermediates O(D^3), no fused-pair materialization
	dL, dp = size(A, 1), size(A, 2)
	dbL, dbR = size(B, 1), size(B, 3)
	dR = size(A, 3)
	dOl, dOr = size(cleft, 1), size(cright, 1)
	mpsj = zeros(T, dOl, dp, dOr)
	Cm = reshape(permutedims(cright, (3, 2, 1)), dbR, dOr * dR)  # (yr, or kr)
	Lm = reshape(cleft, dOl, dL * dbL)                           # (oL, kl yl)
	for p in 1:dp
		W = reshape(@view(B[:, p, :]) * Cm, dbL, dR, dOr)             # (yl, kr, or)
		t1 = @view(A[:, p, :]) * reshape(permutedims(W, (2, 1, 3)), dR, dbL * dOr)
		mpsj[:, p, :] .= reshape(Lm * reshape(t1, dL * dbL, dOr), dOl, dOr)
	end
	return mpsj
end

# ---------- ALS cache ----------

"""
HadamardCache: the ALS "problem" carrier of the iterative hadamard product — find the
chain `omps` maximizing `|⟨omps| ψA ⊙ ψB⟩|` by single-site alternating sweeps.
"""
struct HadamardCache{A, B, O, T}
	ketx::A          # ψA
	kety::B          # ψB
	bra::O           # χ (optimized)
	hstorage::Vector{Array{T,3}}
end

_env_updateleft!(m::HadamardCache, s) =
	(m.hstorage[s+1] = _updateleft(m.hstorage[s], m.bra[s], m.ketx[s], m.kety[s]); m)
_env_updateright!(m::HadamardCache, s) =
	(m.hstorage[s] = _updateright(m.hstorage[s+1], m.bra[s], m.ketx[s], m.kety[s]); m)

function _init_hstorage_right!(m::HadamardCache)
	L = length(m.bra)
	T = scalartype(m.bra)
	m.hstorage[1] = ones(T, 1, 1, 1)
	m.hstorage[L+1] = ones(T, 1, 1, 1)
	for s in L:-1:2
		m.hstorage[s] = _updateright(m.hstorage[s+1], m.bra[s], m.ketx[s], m.kety[s])
	end
	return m
end

"""
	leftsweep!(m::HadamardCache, alg::DMRG1) -> kvals

ALS left sweep of the hadamard problem: at each site the local target is the environment
contraction `Cleft · ψA · ψB · Cright`; the chain is moved by QR and the loss
`norm(target)` is recorded per site. The losses are non-decreasing across the sweep.
"""
function leftsweep!(m::HadamardCache, alg::DMRG1)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		mpsj = _reduce_hadamard_site(m.ketx[s], m.kety[s], m.hstorage[s], m.hstorage[s+1])
		kvals[s] = norm(mpsj)
		(alg.verbosity > 2) && _logupdate(stdout, "l2r", s, kvals[s])
		q, r = _gauge_left(mpsj)
		m.bra[s] = q
		m.bra[s+1] = _contract_first(m.bra[s+1], r)
		_env_updateleft!(m, s)
	end
	mpsj = _reduce_hadamard_site(m.ketx[L], m.kety[L], m.hstorage[L], m.hstorage[L+1])
	kvals[L] = norm(mpsj)
	(alg.verbosity > 2) && _logupdate(stdout, "l2r", L, kvals[L])
	m.bra[L] = mpsj
	return kvals
end

"""
	rightsweep!(m::HadamardCache, alg::DMRG1) -> kvals

ALS right sweep (symmetric to [`leftsweep!`](@ref), LQ gauge moves). `kvals` is ordered by
processing time (sites `L, L-1, …, 1`); the losses are non-decreasing.
"""
function rightsweep!(m::HadamardCache, alg::DMRG1)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		mpsj = _reduce_hadamard_site(m.ketx[s], m.kety[s], m.hstorage[s], m.hstorage[s+1])
		kvals[k] = norm(mpsj)
		(alg.verbosity > 2) && _logupdate(stdout, "r2l", s, kvals[k])
		k += 1
		l, q = _gauge_right(mpsj)
		m.bra[s] = q
		m.bra[s-1] = _contract_last(m.bra[s-1], l)
		_env_updateright!(m, s)
	end
	mpsj = _reduce_hadamard_site(m.ketx[1], m.kety[1], m.hstorage[1], m.hstorage[2])
	kvals[L] = norm(mpsj)
	(alg.verbosity > 2) && _logupdate(stdout, "r2l", 1, kvals[L])
	m.bra[1] = mpsj
	return kvals
end

sweep!(m::HadamardCache, alg::DMRG1) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

# ---------- initial guess & cache construction ----------

"""
	svdguess_hadamard(ψA, ψB, D::Int)

On-the-fly naive SVD of the exact pointwise product as an initial guess for the
iterative hadamard (the exact product itself is never materialized). The bond
dimension is capped at `D` (`truncdim(D)`).
"""
function svdguess_hadamard(ψA, ψB, D::Int)
	# kc[kl, yl, p, k] = Σ_{kr,yr} A[kl,p,kr]·B[yl,p,yr]·c3[kr,yr,k], batched over the
	# physical index (shared elementwise between A and B): no fused-pair materialization
	# (building KB would cost O(D^4·d) memory per site)
	site = (i, carry) -> begin
		A = ψA[i]
		B = ψB[i]
		if carry === nothing
			# the fused pair (aL, bL, p, aR, bR) is never materialized (D^4*d): the
			# tied tensor is built one physical slice at a time (tie's (2,1,2) groups
			# the legs as (aL, bL), (p), (aR, bR) — the slice layout below matches)
			daL, dp, daR = size(A, 1), size(A, 2), size(A, 3)
			dbL, dbR = size(B, 1), size(B, 3)
			kb = zeros(eltype(A), daL * dbL, dp, daR * dbR)
			for p in 1:dp
				kb4 = reshape(@view(kb[:, p, :]), daL, dbL, daR, dbR)
				kb4 .= reshape(@view(A[:, p, :]), daL, 1, daR, 1) .*
					   reshape(@view(B[:, p, :]), 1, dbL, 1, dbR)
			end
			return kb
		end
		daL, dp, daR = size(A, 1), size(A, 2), size(A, 3)
		dbL, dbR = size(B, 1), size(B, 3)
		k = size(carry, 2)
		c3 = reshape(carry, daR, dbR, k)                              # (kr, yr, k)
		kc = zeros(eltype(c3), daL, dbL, dp, k)
		Cm = reshape(c3, daR, dbR * k)                                # (kr, yr k)
		for p in 1:dp
			Z = reshape(@view(A[:, p, :]) * Cm, daL, dbR, k)          # (kl, yr, k)
			Z2 = reshape(permutedims(Z, (2, 1, 3)), dbR, daL * k)     # (yr, kl k)
			res = reshape(@view(B[:, p, :]) * Z2, dbL, daL, k)        # (yl, kl, k)
			kc[:, :, p, :] .= permutedims(res, (2, 1, 3))
		end
		return tie(kc, (2, 1, 1))
	end
	data, _ = _naive_svd_guess(site, length(ψA); trunc=truncdim(D))
	return CanonicalMPS(Vector{Array{scalartype(data[1]),3}}(data))
end

"""
	HadamardCache(ketx, kety, bra)

Build the `HadamardCache` of the iterative hadamard product: allocate the environment
stack and precompute the right environments from the chains as supplied — no
regularization is performed on any chain, and `alg.D` is ignored. Callers provide a
right-canonical `bra` (the `hadamard!` driver never re-gauges or truncates it).
"""
function HadamardCache(ketx, kety, bra)
	T = scalartype(bra)
	cache = HadamardCache(ketx, kety, bra, Vector{Array{T,3}}(undef, length(bra) + 1))
	_init_hstorage_right!(cache)
	# the init gauging rewrites the Schmidt values, and the sweeps never touch them:
	# reset them so "initialized" always implies "properly canonical"
	unset_svectors!(bra)
	return cache
end

"""
	hadamard!(χ, ψA, ψB, alg::DMRG1) -> χ

Single-site variational (ALS) pointwise product of `ψA` and `ψB` refined in place on
the user-supplied initial guess `χ`. The ansatz is taken exactly as supplied — its bond
profile is neither grown nor truncated, and in particular `alg.D` is ignored (it only
sets the bond cap when the guess is drawn automatically by `hadamard`). Supply a
right-canonical `χ`. The sweeps determine direction and magnitude together at the data
level (the exact hadamard product is never formed); the external scale is attached
through the per-site `scaling` field, with the ⊙ convention
`scaling(ψA ⊙ ψB) = scaling(ψA)·scaling(ψB)` — a scaling^L power is never materialized.
"""
function hadamard!(χ, ψA, ψB, alg::DMRG1)
	cache = HadamardCache(ψA, ψB, χ)
	iterative_compute!(cache, alg)
	setscaling!(χ, scaling(ψA) * scaling(ψB))
	return χ
end

# SVD route: exact product then a single compression sweep
_svd_hadamard(ψA, ψB, alg::SVDCompression) =
	_canonicalize!(ψA ⊙ ψB; alg=Orthogonalize(SVD(), alg.trunc, false))

# ---------- exported interface ----------

_validate_hadamard(ψA::CanonicalMPS, ψB::CanonicalMPS) = begin
	(length(ψA) == length(ψB)) || throw(DimensionMismatch("lengths must match"))
	(phydims(ψA) == phydims(ψB)) ||
		throw(DimensionMismatch("physical dimensions must match"))
end

"""
	hadamard(ψA, ψB, alg=SVDCompression(trunc=DefaultTruncation)) -> χ
	hadamard(ψA, ψB, alg::DMRG1) -> χ

Compressed pointwise (Hadamard) product ψA ⊙ ψB: the result is a finite-bond MPS approximation
obtained with `alg` (`SVDCompression`: exact product + one SVD sweep; `DMRG1`: site-by-site
variational ALS on a `svdguess_hadamard(ψA, ψB, alg.D)` initial guess). To refine a custom
ansatz with its own bond profile (ignoring `alg.D` entirely), call
`hadamard!(χ, ψA, ψB, alg)`.

The exact (strict) product without compression is the operator `⊙` (`ψA ⊙ ψB`).
"""
function hadamard(ψA::CanonicalMPS, ψB::CanonicalMPS, alg::SVDCompression=DefaultMultAlg)
	_validate_hadamard(ψA, ψB)
	return _svd_hadamard(ψA, ψB, alg)[1]
end

function hadamard(ψA::CanonicalMPS, ψB::CanonicalMPS, alg::DMRG1)
        _validate_hadamard(ψA, ψB)
        return hadamard!(svdguess_hadamard(ψA, ψB, alg.D), ψA, ψB, alg)
end

# ---------- two-site (DMRG2) sweeps and interface ----------

# the two-site optimal pointwise-product block, batched over the pair of shared physical
# indices: the fused pair of BOTH sites is never materialized (KB2 would hold D^4*d^2
# elements, e.g. 1.07 GiB at D=64, and the n-ary contraction around it produced the same
# order of intermediates); every intermediate below stays at O(D^3)
function _reduce_hadamard_site2(m::HadamardCache, s::Integer)
	A1, A2 = m.ketx[s], m.ketx[s+1]
	B1, B2 = m.kety[s], m.kety[s+1]
	hL = m.hstorage[s]        # (oL, aL, bL)
	hR = m.hstorage[s + 2]    # (oR, aR, bR)
	T = promote_type(eltype(hL), eltype(A1))
	dOL, dAL = size(hL, 1), size(A1, 1)
	dp1, dp2 = size(A1, 2), size(A2, 2)
	dAR, dOR = size(A2, 3), size(hR, 1)
	dbL, dbR = size(B1, 1), size(B2, 3)
	t2 = zeros(T, dOL, dp1, dp2, dOR)
	# the physical pair is shared elementwise between the two kets: broadcast, not contract
	HRm = reshape(permutedims(hR, (3, 2, 1)), dbR, dAR * dOR)   # (bR, aR oR)
	Lm = reshape(hL, dOL, dAL * dbL)                           # (oL, aL bL)
	for p1 in 1:dp1, p2 in 1:dp2
		a2 = @view(A1[:, p1, :]) * @view(A2[:, p2, :])   # (aL, aR)
		b2 = @view(B1[:, p1, :]) * @view(B2[:, p2, :])   # (bL, bR)
		Y = reshape(b2 * HRm, dbL, dAR, dOR)             # (bL, aR, oR)
		Y2 = reshape(permutedims(Y, (2, 1, 3)), dAR, dbL * dOR)   # (aR, bL oR)
		t2[:, p1, p2, :] .= reshape(Lm * reshape(a2 * Y2, dAL * dbL, dOR), dOL, dOR)
	end
	return t2
end

function leftsweep!(m::HadamardCache, alg::DMRG2)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		t2 = _reduce_hadamard_site2(m, s)
		kvals[s] = norm(t2)
		(alg.verbosity > 2) && _logupdate(stdout, "l2r", s, kvals[s])
		_als2_update!(m.bra, s, t2, alg; move_right=true)
		_env_updateleft!(m, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

function rightsweep!(m::HadamardCache, alg::DMRG2)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		t2 = _reduce_hadamard_site2(m, s - 1)
		kvals[k] = norm(t2)
		(alg.verbosity > 2) && _logupdate(stdout, "r2l", s - 1, kvals[k])
		k += 1
		_als2_update!(m.bra, s - 1, t2, alg; move_right=false)
		_env_updateright!(m, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(m::HadamardCache, alg::DMRG2) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

function hadamard!(χ, ψA, ψB, alg::DMRG2)
	cache = HadamardCache(ψA, ψB, χ)
	iterative_compute!(cache, alg)
	# attach the ⊙ convention scale scaling(ψA)·scaling(ψB) and fold the center norm
	# into `scaling` (see `mult!`)
	setscaling!(χ, scaling(ψA) * scaling(ψB))
	_renormalize!(χ, χ[1], false)
	return χ
end

function hadamard(ψA::CanonicalMPS, ψB::CanonicalMPS, alg::DMRG2)
	_validate_hadamard(ψA, ψB)
	return hadamard!(svdguess_hadamard(ψA, ψB, _guess_bond(alg.trunc)), ψA, ψB, alg)
end
