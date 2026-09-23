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
# pointwise-fused local tensor KB[aL,bL,p,aR,bR] = A[aL,p,aR]*B[bL,p,bR]; the physical
# index is NOT summed (a broadcast, not a contraction) — same local map as exact hadamard
function _fused_pair(A::MPSTensor, B::MPSTensor)
	KB = reshape(A, size(A, 1), 1, size(A, 2), size(A, 3), 1) .*
		 reshape(B, 1, size(B, 1), size(B, 2), 1, size(B, 3))
	return KB   # (aL, bL, p, aR, bR)
end

function _updateleft(hold::AbstractArray{T,3}, O::MPSTensor, A::MPSTensor, B::MPSTensor) where {T}
	KB = _fused_pair(A, B)
	@tensor hnew[-1, -2, -3] := conj(O[1, 4, -1]) * hold[1, 2, 3] * KB[2, 3, 4, -2, -3]
	return hnew
end

"""
	_updateright(h, O, A, B) -> h′

Right-to-left transfer of the three-chain environment (symmetric to [`_updateleft`](@ref)).
"""
function _updateright(hold::AbstractArray{T,3}, O::MPSTensor, A::MPSTensor, B::MPSTensor) where {T}
	KB = _fused_pair(A, B)
	@tensor hnew[-1, -2, -3] := conj(O[-1, 4, 1]) * hold[1, 2, 3] * KB[-2, -3, 4, 2, 3]
	return hnew
end

# local ALS target: drive the omps site tensor towards the pointwise product
function _reduce_hadamard_site(A::MPSTensor, B::MPSTensor,
							   cleft::AbstractArray{T,3}, cright::AbstractArray{T,3}) where {T}
	KB = _fused_pair(A, B)
	@tensor mpsj[-1, -2, -3] := cleft[-1, 1, 2] * KB[1, 2, -2, 3, 4] * cright[-3, 3, 4]
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
		q, r = _gauge_left(mpsj)
		m.bra[s] = q
		m.bra[s+1] = _contract_first(m.bra[s+1], r)
		_env_updateleft!(m, s)
	end
	mpsj = _reduce_hadamard_site(m.ketx[L], m.kety[L], m.hstorage[L], m.hstorage[L+1])
	kvals[L] = norm(mpsj)
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
		k += 1
		l, q = _gauge_right(mpsj)
		m.bra[s] = q
		m.bra[s-1] = _contract_last(m.bra[s-1], l)
		_env_updateright!(m, s)
	end
	mpsj = _reduce_hadamard_site(m.ketx[1], m.kety[1], m.hstorage[1], m.hstorage[2])
	kvals[L] = norm(mpsj)
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
	site = i -> begin
		A = ψA[i]
		B = ψB[i]
		r = reshape(A, size(A, 1), 1, size(A, 2), size(A, 3), 1) .*
			reshape(B, 1, size(B, 1), size(B, 2), 1, size(B, 3))
		tie(r, (2, 1, 2))
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

# the two-site optimal pointwise-product block: the fused pairs of both sites
function _reduce_hadamard_site2(m::HadamardCache, s::Integer)
	A2 = @tensor a[aL, p1, p2, aR] := m.ketx[s][aL, p1, b] * m.ketx[s+1][b, p2, aR]
	B2 = @tensor b[bL, q1, q2, bR] := m.kety[s][bL, q1, c] * m.kety[s+1][c, q2, bR]
	# the pointwise product shares ONE physical value per site: broadcast, not contract
	KB2 = reshape(A2, size(A2, 1), 1, size(A2, 2), size(A2, 3), size(A2, 4), 1) .*
		  reshape(B2, 1, size(B2, 1), size(B2, 2), size(B2, 3), 1, size(B2, 4))
	# KB2: (aL, bL, p1, p2, aR, bR)
	return @tensor t[-1, -2, -3, -4] := m.hstorage[s][-1, 1, 2] * KB2[1, 2, -2, -3, 3, 4] *
									   m.hstorage[s+2][-4, 3, 4]
end

function leftsweep!(m::HadamardCache, alg::DMRG2)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		t2 = _reduce_hadamard_site2(m, s)
		kvals[s] = norm(t2)
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
