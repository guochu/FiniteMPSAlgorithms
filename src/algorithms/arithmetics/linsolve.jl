# iterative (ALS) linear solve: given an MPO A and an MPS y, find a finite-bond MPS x
# minimizing ||A·x - y||². The stationarity condition is the normal equation
# A†A·x = A†y, so two environment stacks are swept simultaneously:
#   hstorage: ⟨x| A†A |x⟩  (four indices: xL, aL, aR, xR)
#   bstorage: ⟨x| A†  |y⟩  (three indices: xL, a, yR)
# Template mirrors mult.jl; local solves are dense (small single-site systems).

# ---------- environment transfer primitives ----------
# A site tensor is W[aL, po, aR, pi]; A† element = conj(W[aRdag, po, aLdag, pi]).
# Careful: the bra physical (A† output, 4th slot) differs from A's output (2nd slot),
# so every index appears at most twice within one contraction.

"""
	updateleft!(::Array{<:Any,4}, x, W, x) -> h′

⟨x|A†A|x⟩ left transfer; `h::Array{T,4}` axes (xL, A†-bond, A-bond, xR).
"""
function _h_updateleft(hold::AbstractArray{T,4},
					   x::MPSTensor, W::MPOTensor, xk::MPSTensor) where {T}
	@tensor hnew[-1, -2, -3, -4] :=
		conj(x[1, pb, -1]) * conj(W[2, po, -2, pb]) * hold[1, 2, 3, 4] *
		W[3, po, -3, pi] * xk[4, pi, -4]
	return hnew
end

function _h_updateright(hold::AbstractArray{T,4},
						x::MPSTensor, W::MPOTensor, xk::MPSTensor) where {T}
	@tensor hnew[-1, -2, -3, -4] :=
		conj(x[-1, pb, 1]) * conj(W[-2, po, 2, pb]) * hold[1, 2, 3, 4] *
		W[-3, po, 3, pi] * xk[-4, pi, 4]
	return hnew
end

"""
	_b_updateleft(::Array{<:Any,3}, x, W, y) -> b′

⟨x|A†|y⟩ left transfer; `b::Array{T,3}` axes (xL, A†-bond, yR).
"""
function _b_updateleft(hold::AbstractArray{T,3},
					   x::MPSTensor, W::MPOTensor, y::MPSTensor) where {T}
	@tensor bnew[-1, -2, -3] :=
		conj(x[1, pb, -1]) * conj(W[2, py, -2, pb]) * hold[1, 2, 3] * y[3, py, -3]
	return bnew
end

function _b_updateright(hold::AbstractArray{T,3},
					   x::MPSTensor, W::MPOTensor, y::MPSTensor) where {T}
	@tensor bnew[-1, -2, -3] :=
		conj(x[-1, pb, 1]) * conj(W[-2, py, 2, pb]) * hold[1, 2, 3] * y[-3, py, 3]
	return bnew
end

# ---------- local normal equation at one site ----------

# H_eff z = (A†A)_eff z; t_eff = (A†y)_eff
function _h_apply(z::MPSTensor, W::MPOTensor,
				  hL::AbstractArray{T,4}, hR::AbstractArray{T,4}) where {T}
	@tensor y[-1, -2, -3] :=
		hL[-1, cL, aL, 1] * conj(W[cL, po, cR, -2]) *
		W[aL, po, kR, pi] * z[1, pi, 4] * hR[-3, cR, kR, 4]
	return y
end

function _b_target(W::MPOTensor, y::MPSTensor,
				   bL::AbstractArray{T,3}, bR::AbstractArray{T,3}) where {T}
	@tensor t[-1, -2, -3] :=
		bL[-1, cL, 1] * conj(W[cL, py, cR, -2]) * y[1, py, 2] * bR[-3, cR, 2]
	return t
end

# ---------- ALS cache ----------

"""
LinsolveCache: ALS problem carrier for [`linsolve`](@ref) — find x minimizing ||A·x - y||²
with single-site sweeps over the two (A†A and A†) environment stacks.
"""
struct LinsolveCache{A, Y, X, T}
	mpo::A
	bra::Y
	ket::X
	hstorage::Vector{Array{T,4}}
	bstorage::Vector{Array{T,3}}
end

_h_left!(m::LinsolveCache, s) =
	(m.hstorage[s+1] = _h_updateleft(m.hstorage[s], m.ket[s], m.mpo[s], m.ket[s]); m)
_h_right!(m::LinsolveCache, s) =
	(m.hstorage[s] = _h_updateright(m.hstorage[s+1], m.ket[s], m.mpo[s], m.ket[s]); m)
_b_left!(m::LinsolveCache, s) =
	(m.bstorage[s+1] = _b_updateleft(m.bstorage[s], m.ket[s], m.mpo[s], m.bra[s]); m)
_b_right!(m::LinsolveCache, s) =
	(m.bstorage[s] = _b_updateright(m.bstorage[s+1], m.ket[s], m.mpo[s], m.bra[s]); m)

function _init_storages_right!(m::LinsolveCache)
	L = length(m.ket)
	rightorth!(m.ket)
	T = scalartype(m.bra)
	m.hstorage[1] = ones(T, 1, 1, 1, 1)
	m.hstorage[L+1] = ones(T, 1, 1, 1, 1)
	m.bstorage[1] = ones(T, 1, 1, 1)
	m.bstorage[L+1] = ones(T, 1, 1, 1)
	for s in L:-1:2
		m.hstorage[s] = _h_updateright(m.hstorage[s+1], m.ket[s], m.mpo[s], m.ket[s])
		m.bstorage[s] = _b_updateright(m.bstorage[s+1], m.ket[s], m.mpo[s], m.bra[s])
	end
	return m
end

function _site_solve(m::LinsolveCache, s::Integer)
	W = m.mpo[s]
	t = _b_target(W, m.bra[s], m.bstorage[s], m.bstorage[s+1])
	shape = (size(m.hstorage[s], 1), size(W, 2), size(m.hstorage[s+1], 1))
	n = prod(shape)
	z0 = m.ket[s]
	if size(z0) != shape
		z0 = zeros(scalartype(m.bra), shape)
	end
	Hmat = zeros(promote_type(scalartype(z0), scalartype(W)), n, n)
	E = zeros(scalartype(Hmat), shape)
	for i in 1:n
		fill!(E, 0)
		E[i] = 1
		Hmat[:, i] .= vec(_h_apply(E, W, m.hstorage[s], m.hstorage[s+1]))
	end
	return reshape(Hmat \ vec(t), shape)
end

"""
	leftsweep!(m::LinsolveCache, alg) -> kvals

One left-to-right ALS sweep: at each site the local normal equation H z = t is solved,
the chain is moved by QR and both environment stacks incremented.
"""
function leftsweep!(m::LinsolveCache, alg::DMRGAlgorithm)
	L = length(m.ket)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		z = _site_solve(m, s)
		kvals[s] = norm(z)
		q, r = _gauge_left(z)
		m.ket[s] = q
		m.ket[s+1] = _contract_first(m.ket[s+1], r)
		_h_left!(m, s)
		_b_left!(m, s)
	end
	z = _site_solve(m, L)
	kvals[L] = norm(z)
	m.ket[L] = z
	return kvals
end

"""
	rightsweep!(m::LinsolveCache, alg) -> kvals

One right-to-left ALS sweep (symmetric, LQ gauge moves). `kvals` is ordered by processing
time — sites `L, L-1, …, 1`.
"""
function rightsweep!(m::LinsolveCache, alg::DMRGAlgorithm)
	L = length(m.ket)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		z = _site_solve(m, s)
		kvals[k] = norm(z)
		k += 1
		l, q = _gauge_right(z)
		m.ket[s] = q
		m.ket[s-1] = _contract_last(m.ket[s-1], l)
		_h_right!(m, s)
		_b_right!(m, s)
	end
	z = _site_solve(m, 1)
	kvals[L] = norm(z)
	m.ket[1] = z
	return kvals
end

sweep!(m::LinsolveCache, alg::DMRGAlgorithm) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

# ---------- global residual ||A·x - y|| ----------

function _residual_norm(m::LinsolveCache)
	T = scalartype(m.bra)
	L = length(m.ket)
	h = ones(T, 1, 1, 1, 1)
	for s in 1:L
		h = _h_updateleft(h, m.ket[s], m.mpo[s], m.ket[s])
	end
	t1 = real(h[1, 1, 1, 1])
	b = ones(T, 1, 1, 1)
	for s in 1:L
		b = _b_updateleft(b, m.ket[s], m.mpo[s], m.bra[s])
	end
	t2 = real(b[1, 1])
	r2 = t1 - 2 * t2 + real(_dot(m.bra, m.bra))
	return sqrt(abs(r2))
end

# initial guesses: `randommps(T, ophydims(A); D=D)` or a compressed right-hand side
# `svdguess_compress(y, trunc)` — the generic initializers cover the linsolve case

"""
	LinsolveCache(A, y, x)

Build the `LinsolveCache` of the iterative `linsolve`: allocate the h/b environment
stacks and initialize them by a data-level right-orthogonalization of `x`.
"""
function LinsolveCache(A, y, x)
	T = promote_type(scalartype(A), scalartype(y))
	m = LinsolveCache(A, y, x,
					  Vector{Array{T,4}}(undef, length(y) + 1),
					  Vector{Array{T,3}}(undef, length(y) + 1))
	_init_storages_right!(m)
	return m
end

"""
	linsolve!(x, A, y, alg::DMRG1) -> x

Single-site variational (ALS) solve of `A·x ≈ y` refined in place on the initial guess
`x`. The residual converges with `|r_n - r_{n-1}| < alg.tol`. The solution is
expressed in the right-hand side's per-site scaling convention (a scaling^L power is
never materialized).
"""
function linsolve!(x, A, y, alg::DMRG1)
	m = LinsolveCache(A, y, x)
	prev = Inf
	for _ in 1:alg.maxiter
		sweep!(m, alg)
		r = _residual_norm(m)
		abs(r - prev) < alg.tol && break
		prev = r
	end
	# the ALS reads raw data only: the solution is expressed in the right-hand side's
	# per-site scaling convention (attached through the `scaling` field — a scaling^L
	# power is never materialized)
	setscaling!(m.ket, scaling(y))
	return x
end

# ---------- exported interface ----------

function _validate_linsolve(A::AbstractMPO, y::CanonicalMPS)
	(length(A) == length(y)) || throw(DimensionMismatch("lengths must match"))
	(ophydims(A) == phydims(y)) ||
		throw(DimensionMismatch("physical dimensions must match"))
	return nothing
end

"""
	linsolve(A, y, alg=DMRG1(); D=Defaults.D) -> x

Solve `A·x ≈ y` variationally: find a finite-bond MPS `x` of bond dimension `D`
minimizing the residual norm `||A·x - y||` by single-site ALS sweeps (normal equation
A†A x = A† y). The initial guess is a random state. Block-sparse (Hamiltonian) inputs
are expanded to the dense MPO layer.
"""
function linsolve(A::AbstractMPO, y::CanonicalMPS, alg::DMRG1=DMRG1(); D::Int=Defaults.D)
	_validate_linsolve(A, y)
	T = promote_type(scalartype(A), scalartype(y))
	x = randommps(T, ophydims(A); D, normalize=false)
	return linsolve!(x, A, y, alg)
end

linsolve(A::MPOHamiltonian, y::CanonicalMPS, alg::DMRG1=DMRG1(); D::Int=Defaults.D) =
	linsolve(MPO(tompotensors(A)), y, alg; D)
