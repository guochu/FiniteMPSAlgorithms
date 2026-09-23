# mult: the unique exported interface for (truncating) operator application
#   mult(h, ψ, alg)   MPO·MPS          -> CanonicalMPS
#   mult(hA, hB, alg) MPO·MPO          -> CanonicalMPO
# (hB may be a density matrix / process tensor; the external scales of canonical
#  operands are inherited through the output's `scaling` field)
# dispatched on alg: SVDCompression (exact product + SVD-sweep compression) or
# DMRG1 (single-site variational ALS sweeping, following TEMPO's iterativemult)

# ---------- small generic helpers for bra chains of rank 3/4 ----------

_contract_first(B::MPSTensor, r::AbstractMatrix) = @tensor z[-1, -2, -3] := r[-1, 1] * B[1, -2, -3]
_contract_first(B::MPOTensor, r::AbstractMatrix) = @tensor z[-1, -2, -3, -4] := r[-1, 1] * B[1, -2, -3, -4]
_contract_last(A::MPSTensor, l::AbstractMatrix) = @tensor z[-1, -2, -3] := A[-1, -2, 1] * l[1, -3]
_contract_last(A::MPOTensor, l::AbstractMatrix) = @tensor z[-1, -2, -3, -4] := A[-1, -2, 1, -4] * l[1, -3]

_gauge_left(A::MPSTensor) = leftorth!(A, (1, 2), (3,))
function _gauge_left(A::MPOTensor)
	q, r = leftorth!(A, (1, 2, 4), (3,))
	return permute(q, (1, 2, 4, 3)), r
end
_gauge_right(A::MPSTensor) = rightorth!(A, (1,), (2, 3))
_gauge_right(A::MPOTensor) = rightorth!(A, (1,), (2, 3, 4))

# ---------- on-the-fly naive SVD initial guess (shared by mult / add / hadamard) ----------
# Stream the exact chain's site tensors (generated lazily by `site(i)` for i = L:-1:1)
# right to left; each is right-orthogonalized with truncation under `trunc` (bond ≤ D):
# the right factor becomes the site tensor and the left factor times the spectrum is
# carried into the next site. The exact chain itself is never materialized. Returns a
# right-canonical chain (vectors of rank-3/rank-4 tensors) with the remainder (scale)
# absorbed into site 1, plus the maximal truncation error.
function _naive_svd_guess(site::Function, L::Int; trunc::TruncationScheme)
	carry = nothing
	proto = copy(site(L))
	out = Vector{typeof(proto)}(undef, L)
	maxerr = 0.0
	for i in L:-1:2
		B = i == L ? proto : copy(site(i))
		carry === nothing || (B = _contract_last(B, carry))
		N = ndims(B)
		u, s, v, err = tsvd!(B, (1,), ntuple(d -> d + 1, Val(N - 1)); trunc)
		out[i] = v
		carry = u * Diagonal(s)
		maxerr = max(maxerr, err)
	end
	out[1] = carry === nothing ? copy(site(1)) : _contract_last(copy(site(1)), carry)
	return out, maxerr
end

# three-chain MPO·MPO environment transfers ⟨ompo|mpo|impo⟩
function _updateleft3(hold::AbstractArray{T,3}, O::MPOTensor, M::MPOTensor, I::MPOTensor) where {T}
	@tensor hnew[-1, -2, -3] := conj(O[1, 4, -1, 6]) * hold[1, 2, 3] * M[2, 4, -2, 5] * I[3, 5, -3, 6]
	return hnew
end
function _updateright3(hold::AbstractArray{T,3}, O::MPOTensor, M::MPOTensor, I::MPOTensor) where {T}
	@tensor hnew[-1, -2, -3] := conj(O[-1, 4, 1, 6]) * hold[1, 2, 3] * M[-2, 4, 2, 5] * I[-3, 5, 3, 6]
	return hnew
end

_env_updateleft(h, O, M, B::MPSTensor) = _updateleft(h, O, M, B)
_env_updateleft(h, O, M, B::MPOTensor) = _updateleft3(h, O, M, B)
_env_updateright(h, O, M, B::MPSTensor) = _updateright(h, O, M, B)
_env_updateright(h, O, M, B::MPOTensor) = _updateright3(h, O, M, B)

# local ALS targets
function _reduce_site(A::MPSTensor, X::MPOTensor, cleft, cright)
	@tensor tmp[-1, -2, -3] := cleft[-1, 1, 2] * A[2, 3, 4] * X[1, -2, 5, 3] * cright[-3, 5, 4]
	return tmp
end
function _reduce_site(A::MPSTensor, X::AbstractSparseMPOTensor, cleft, cright)
	T = promote_type(scalartype(A), scalartype(cleft), scalartype(X))
	tmp = zeros(T, size(cleft, 1), phydim(X), size(cright, 1))
	for (wL, wR) in keys(X)
		O = X[wL, wR]
		hl = cleft[:, wL, :]
		hr = cright[:, wR, :]
		@tensor tn[a, p, c] := hl[a, bL] * A[bL, q, bR] * O[p, q] * hr[c, bR]
		tmp .+= tn
	end
	return tmp
end
function _reduce_site(A::MPOTensor, X::MPOTensor, cleft, cright)
	@tensor tmp[-1, -2, -3, -4] := cleft[-1, 1, 2] * A[2, 3, 4, -4] * X[1, -2, 5, 3] * cright[-3, 5, 4]
	return tmp
end

# ---------- ALS cache ----------

"""
MultCache: the ALS "problem" carrier of `mult` — find the bra chain `bra` maximizing
`|⟨bra| H |ket⟩|` (single-site alternating sweeps).
"""
struct MultCache{M, I, O, T}
	H::M
	ket::I
	bra::O
	hstorage::Vector{Array{T,3}}
end

_mult_leftbound(m, T) = m.H isa MPOHamiltonian ? l_LL(m.bra, m.H, m.ket) : ones(T, 1, 1, 1)
_mult_rightbound(m, T) = m.H isa MPOHamiltonian ? r_RR(m.bra, m.H, m.ket) : ones(T, 1, 1, 1)

function _init_hstorage_right!(m::MultCache)
	L = length(m.bra)
	T = scalartype(m.bra)
	m.hstorage[1] = _mult_leftbound(m, T)
	m.hstorage[L+1] = _mult_rightbound(m, T)
	for s in L:-1:2
		m.hstorage[s] = _env_updateright(m.hstorage[s+1], m.bra[s], m.H[s], m.ket[s])
	end
	return m
end

_env_updateleft!(m::MultCache, s) = (m.hstorage[s+1] = _env_updateleft(m.hstorage[s], m.bra[s], m.H[s], m.ket[s]); m)
_env_updateright!(m::MultCache, s) = (m.hstorage[s] = _env_updateright(m.hstorage[s+1], m.bra[s], m.H[s], m.ket[s]); m)

"""
	leftsweep!(m::MultCache, alg::DMRG1) -> kvals

ALS left sweep for the `mult` problem: the local target at each site is the environment
contraction `Cleft · imps · mpo · Cright`; the bra is moved by QR and the loss
`norm(target)` is recorded per site.
"""
function leftsweep!(m::MultCache, alg::DMRG1)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		mpsj = _reduce_site(m.ket[s], m.H[s], m.hstorage[s], m.hstorage[s+1])
		kvals[s] = norm(mpsj)
		q, r = _gauge_left(mpsj)
		m.bra[s] = q
		m.bra[s+1] = _contract_first(m.bra[s+1], r)
		_env_updateleft!(m, s)
	end
	mpsj = _reduce_site(m.ket[L], m.H[L], m.hstorage[L], m.hstorage[L+1])
	kvals[L] = norm(mpsj)
	m.bra[L] = mpsj
	return kvals
end

"""
	rightsweep!(m::MultCache, alg::DMRG1) -> kvals

ALS right sweep for the `mult` problem (symmetric to `leftsweep!`, LQ gauge moves).
`kvals` is ordered by processing time — sites `L, L-1, …, 1`.
"""
function rightsweep!(m::MultCache, alg::DMRG1)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		mpsj = _reduce_site(m.ket[s], m.H[s], m.hstorage[s], m.hstorage[s+1])
		kvals[k] = norm(mpsj)
		k += 1
		l, q = _gauge_right(mpsj)
		m.bra[s] = q
		m.bra[s-1] = _contract_last(m.bra[s-1], l)
		_env_updateright!(m, s)
	end
	mpsj = _reduce_site(m.ket[1], m.H[1], m.hstorage[1], m.hstorage[2])
	kvals[L] = norm(mpsj)
	m.bra[1] = mpsj
	return kvals
end

sweep!(m::MultCache, alg::DMRG1) = begin
	k1 = leftsweep!(m, alg)
	k2 = rightsweep!(m, alg)
	return vcat(k1, k2)
end

# ---------- SVD route & naive exact-product guesses ----------

function _svd_mult(h::AbstractMPO, x, alg::SVDCompression)
	# `h * x` is scaling-aware: the external per-site scales of canonical operands are
	# already attached to the product's `scaling` field (a scaling^L power is never
	# materialized), and the canonicalization keeps the represented value by rescaling
	# the data into that field. Only a scale-free plain-MPO product is wrapped.
	out = h * x
	chain, err = _canonicalize!(out; alg=Orthogonalize(SVD(), alg.trunc, false))
	chain isa Union{CanonicalMPS, CanonicalMPO} && return chain, err
	return CanonicalMPO(chain.data), err
end

# naive on-the-fly SVD of the exact product: the site tensors of `h·x` are formed one at
# a time and streamed through a truncating right-orthogonalization (never materializing
# the full exact product)
_naive_svd_mult(h::MPOHamiltonian, x::CanonicalMPS, trunc::TruncationScheme) =
	_naive_svd_mult(MPO(tompotensors(h)), x, trunc)
_naive_svd_mult(h::MPOHamiltonian, x::AbstractMPO, trunc::TruncationScheme) =
	_naive_svd_mult(MPO(tompotensors(h)), x, trunc)
function _naive_svd_mult(h::AbstractMPO, x::CanonicalMPS, trunc::TruncationScheme)
	site = i -> begin
		W = h[i]
		A = x[i]
		@tensor r[aL, po, aR, bL, bR] := W[aL, po, aR, pin] * A[bL, pin, bR]
		tie(permute(r, (1, 4, 2, 3, 5)), (2, 1, 2))
	end
	return _naive_svd_guess(site, length(h); trunc)
end
function _naive_svd_mult(h::AbstractMPO, x::AbstractMPO, trunc::TruncationScheme)
	site = i -> begin
		WA = h[i]
		WB = x[i]
		@tensor t[aA, poA, aA2, aB, aB2, q] := WA[aA, poA, aA2, p] * WB[aB, p, aB2, q]
		tie(permute(t, (1, 4, 2, 3, 5, 6)), (2, 1, 2, 1))
	end
	return _naive_svd_guess(site, length(h); trunc)
end

_wrap_mult_guess(data, ::CanonicalMPS) = CanonicalMPS(Vector{Array{scalartype(data[1]),3}}(data))
_wrap_mult_guess(data, ::AbstractMPO) = CanonicalMPO(Vector{Array{scalartype(data[1]),4}}(data))

function _random_output(x::CanonicalMPS, ds_out, D)
	return randommps(scalartype(x), ds_out; D)
end
function _random_output(x::AbstractMPO, ds_out, ds_in, D)
	L = length(x)
	T = scalartype(x)
	prof = max_bonddims(max.(ds_out, ds_in), D)
	data = Vector{Array{T,4}}(undef, L)
	for i in 1:L
		dl = i == 1 ? 1 : prof[i]
		dr = i == L ? 1 : prof[i+1]
		data[i] = randn(T, dl, ds_out[i], dr, ds_in[i])
	end
	out = CanonicalMPO(data)
	canonicalize!(out)
	return out
end

# ---------- initial guess & cache construction ----------

"""
	svdguess_mult(h, x, D::Int)

On-the-fly naive SVD of the exact product h·x as an initial guess for the iterative
`mult`: the site tensors of h·x are formed one at a time and streamed through a
truncating right-orthogonalization with bond cap `D` (`truncdim(D)`; the exact product
itself is never materialized).
"""
function svdguess_mult(h, x, D::Int)
	data, _ = _naive_svd_mult(h, x, truncdim(D))
	return _wrap_mult_guess(data, x)
end
# block-sparse operands are expanded into the dense MPO layer
svdguess_mult(h::MPOHamiltonian, x, D::Int) =
	svdguess_mult(MPO(tompotensors(h)), x, D)
svdguess_mult(h::AbstractMPO, x::MPOHamiltonian, D::Int) =
	svdguess_mult(h, MPO(tompotensors(x)), D)
svdguess_mult(h::MPOHamiltonian, x::MPOHamiltonian, D::Int) =
	svdguess_mult(MPO(tompotensors(h)), MPO(tompotensors(x)), D)

"""
	MultCache(h, x, bra)

Build the `MultCache` of the iterative `mult`: allocate the environment stack and
precompute the right environments from the chains as supplied — no regularization is
performed on any chain, and `alg.D` is ignored. Callers provide a right-canonical `bra`
(the `mult!` driver never re-gauges or truncates it).
"""
function MultCache(h, x, bra)
	T = promote_type(scalartype(h), scalartype(x))
	cache = MultCache(h, x, bra, Vector{Array{T,3}}(undef, length(x) + 1))
	_init_hstorage_right!(cache)
	# the init gauging rewrites the Schmidt values, and the sweeps never touch them:
	# reset them so "initialized" always implies "properly canonical"
	unset_svectors!(bra)
	return cache
end

"""
	mult!(out, h, x, alg::DMRG1) -> out

Single-site variational (ALS) evaluation of h·x refined in place on the user-supplied
initial guess `out`. The ansatz is taken exactly as supplied — its bond profile is
neither grown nor truncated, and in particular `alg.D` is ignored (it only sets the bond
cap when the guess is drawn automatically by `mult`). Supply a right-canonical `out`.
The sweeps determine direction and magnitude together at the data level (the exact
product is never formed); the external scales of canonical operands are attached through
the per-site `scaling` field of the output — a scaling^L power is never materialized.
"""
function mult!(out, h, x, alg::DMRG1)
	cache = MultCache(h, x, out)
	iterative_compute!(cache, alg)
	s = _opscaling(h) * _opscaling(x)
	s == 1 || setscaling!(out, s)
	return out
end

# the external per-site scale of a canonical operand lives in its `scaling` field; plain
# MPO chains keep their scale in the data
_opscaling(x::AbstractMPO) = 1.0
_opscaling(x::Union{CanonicalMPS, CanonicalMPO}) = scaling(x)

# ---------- exported interface ----------

"""
	mult(h, x, alg=SVDCompression(trunc=DefaultTruncation)) -> CanonicalMPO

Apply the operator `h` to the state / operator `x` (SVD route): the exact product is
formed and compressed by a single SVD sweep under `alg.trunc`. For the variational ALS
route supply a bond cap: `mult(h, x, DMRG1(; ...); D=D)` — the initial guess is drawn
by `svdguess_mult(h, x, D)`.
"""
function mult(h::AbstractMPO, x, alg::SVDCompression=DefaultMultAlg)
	(length(h) == length(x)) || throw(ArgumentError("dimension mismatch"))
	return _svd_mult(h, x, alg)[1]
end

"""
	mult(h, x, alg::DMRG1) -> CanonicalMPO

Variational (ALS) evaluation of h·x with bond cap `alg.D`: the initial guess is drawn by
`svdguess_mult(h, x, alg.D)` and refined by sweeps. The operands may be static `MPO`s or
canonical chains (density matrices / process tensors); the external per-site scales of
canonical operands are inherited through the output's `scaling` field (never
materialized as a scaling^L power). The result is always a `CanonicalMPO`. To refine a
custom ansatz with its own bond profile (ignoring `alg.D` entirely), call
`mult!(out, h, x, alg)`.
"""
function mult(h::AbstractMPO, x, alg::DMRG1)
	(length(h) == length(x)) || throw(ArgumentError("dimension mismatch"))
	return mult!(svdguess_mult(h, x, alg.D), h, x, alg)
end

# block-sparse operands are expanded into the dense MPO layer (the three-chain ALS /
# exact products are written on 4-index tensors)
mult(hA::MPOHamiltonian, x, alg::SVDCompression=DefaultMultAlg) =
	mult(MPO(tompotensors(hA)), x, alg)
mult(hA::MPOHamiltonian, x, alg::DMRG1) =
	mult(MPO(tompotensors(hA)), x, alg)
mult(h::AbstractMPO, hB::MPOHamiltonian, alg::SVDCompression=DefaultMultAlg) =
	mult(h, MPO(tompotensors(hB)), alg)
mult(h::AbstractMPO, hB::MPOHamiltonian, alg::DMRG1) =
	mult(h, MPO(tompotensors(hB)), alg)
mult(hA::MPOHamiltonian, hB::MPOHamiltonian, alg::SVDCompression=DefaultMultAlg) =
	mult(MPO(tompotensors(hA)), MPO(tompotensors(hB)), alg)
mult(hA::MPOHamiltonian, hB::MPOHamiltonian, alg::DMRG1) =
	mult(MPO(tompotensors(hA)), MPO(tompotensors(hB)), alg)

# `hA * hB` contracts the raw data; a canonical operand keeps its per-site scale in the
# `scaling` field, so the wrapped CanonicalMPO inherits that factor through its own
# `scaling` field (attached by the caller, see `_opscaling`). No re-canonicalization:
# the data keeps the full scale (consistent with the DMRG1 route), the Schmidt values
# are initialized lazily.
_wrap_canonicalmpo(out::CanonicalMPO) = out
_wrap_canonicalmpo(out::AbstractMPO) = CanonicalMPO(out.data)

# ---------- two-site (DMRG2) sweeps and interface ----------

# the two-site optimal block: left environment · ket pair · MPO pair · right environment
function _reduce_site2(m::MultCache, s::Integer)
	cL, cR = m.hstorage[s], m.hstorage[s+2]
	W2 = @tensor W[u, q1, q2, v, r1, r2] := m.H[s][u, q1, c, r1] * m.H[s+1][c, q2, v, r2]
	if m.ket[s] isa MPSTensor
		k2 = @tensor k[a, q1, q2, b] := m.ket[s][a, q1, c] * m.ket[s+1][c, q2, b]
		return @tensor t[-1, -2, -3, -4] := cL[-1, 1, 2] * k2[2, q1, q2, 3] *
										   W2[1, -2, -3, 4, q1, q2] * cR[-4, 4, 3]
	end
	# explicit binary contraction steps (the n-ary tree is unreliable at 6 free legs).
	# W2 legs: (wL, po1, po2, wR, pi1, pi2); the ket's OUT legs contract W2's IN legs
	# (the H·ket matrix product, matching `_updateleft3`), the bra's free pair is
	# (W2.po, ket.pi)
	k2 = @tensor k[a, o1, o2, b, i1, i2] := m.ket[s][a, o1, c, i1] * m.ket[s+1][c, o2, b, i2]
	hh = @tensor hh[b, w1, o1, o2, bR, i1, i2] := cL[b, w1, kL] * k2[kL, o1, o2, bR, i1, i2]
	hw = @tensor hw[b, p1, p2, w2, bR, i1, i2] := hh[b, w1, o1, o2, bR, i1, i2] *
												 W2[w1, p1, p2, w2, o1, o2]
	return @tensor t[b, -2, -3, -4, -5, -6] := hw[b, -2, -3, w2, bR, -4, -5] *
											   cR[-6, w2, bR]
end

function leftsweep!(m::MultCache, alg::DMRG2)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		t2 = _reduce_site2(m, s)
		kvals[s] = norm(t2)
		_als2_update!(m.bra, s, t2, alg; move_right=true)
		# hstorage[s+1] = left env over sites 1..s — exactly the next pair's left env;
		# the right envs hstorage[s+2..] must stay untouched
		_env_updateleft!(m, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

function rightsweep!(m::MultCache, alg::DMRG2)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		t2 = _reduce_site2(m, s - 1)
		kvals[k] = norm(t2)
		k += 1
		_als2_update!(m.bra, s - 1, t2, alg; move_right=false)
		_env_updateright!(m, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(m::MultCache, alg::DMRG2) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

# the initial guesses are drawn with the bond cap carried by `alg.trunc`
# (`_guess_bond`); the two-site sweeps adapt the bond dimension during the iterations,
# so the guess may be smaller than the final bond
function mult!(out, h, x, alg::DMRG2)
	cache = MultCache(h, x, out)
	iterative_compute!(cache, alg)
	# attach the external operand scales and fold the center tensor's norm (the total
	# data norm — the swept chain is isometric on the other sites) into the `scaling`
	# field; the bond spectra recorded during the sweeps initialize the Schmidt values
	s = _opscaling(h) * _opscaling(x)
	s == 1 || setscaling!(out, s)
	_renormalize!(out, out[1], false)
	return out
end

function mult(h::AbstractMPO, x, alg::DMRG2)
	(length(h) == length(x)) || throw(ArgumentError("dimension mismatch"))
	return mult!(svdguess_mult(h, x, _guess_bond(alg.trunc)), h, x, alg)
end
mult(hA::MPOHamiltonian, x, alg::DMRG2) = mult(MPO(tompotensors(hA)), x, alg)
mult(h::AbstractMPO, hB::MPOHamiltonian, alg::DMRG2) = mult(h, MPO(tompotensors(hB)), alg)
mult(hA::MPOHamiltonian, hB::MPOHamiltonian, alg::DMRG2) =
	mult(MPO(tompotensors(hA)), MPO(tompotensors(hB)), alg)
