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
		W[3, po, -3, pin] * xk[4, pin, -4]
	return hnew
end

function _h_updateright(hold::AbstractArray{T,4},
						x::MPSTensor, W::MPOTensor, xk::MPSTensor) where {T}
	@tensor hnew[-1, -2, -3, -4] :=
		conj(x[-1, pb, 1]) * conj(W[-2, po, 2, pb]) * hold[1, 2, 3, 4] *
		W[-3, po, 3, pin] * xk[-4, pin, 4]
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
		W[aL, po, kR, pin] * z[1, pin, 4] * hR[-3, cR, kR, 4]
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
LinsolveCache: ALS problem carrier for [`linsolve`](@ref) — find x minimizing ||A·x − y||²
with single-site sweeps over the two (A†A and A†) environment stacks. `ynorm2` is the
raw-data ‖y‖², the constant of the global objective.
"""
struct LinsolveCache{A, Y, X, T}
	mpo::A
	bra::Y
	ket::X
	hstorage::Vector{Array{T,4}}
	bstorage::Vector{Array{T,3}}
	ynorm2::Float64
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

function _site_solve(m::LinsolveCache, s::Integer, solver::KrylovKit.LinearSolver)
        W = m.mpo[s]
        t = _b_target(W, m.bra[s], m.bstorage[s], m.bstorage[s+1])
        shape = (size(m.hstorage[s], 1), size(W, 2), size(m.hstorage[s+1], 1))
        z0 = size(m.ket[s]) == shape ? m.ket[s] : zero(t)
        # the local normal equation is solved iteratively (KrylovKit) on the linear
        # operator action — the dense H matrix is never formed
        z, _ = KrylovKit.linsolve(y -> _h_apply(y, W, m.hstorage[s], m.hstorage[s+1]), t, z0, solver)
        return z, t
end

# the local-solve algorithm of the sweep: the linsolve algorithm types carry their own
# solver, the other iterative algorithms use the package default
_solverof(alg::ALSLinSolve) = alg.solver
_solverof(alg::ALSLinSolve2) = alg.solver
_solverof(alg) = DefaultLinearSolver

# the exact global residual² ‖A·x − y‖², evaluated from the local decomposition at the
# site with the updated tensor z — no global contraction is needed
function _site_loss(m::LinsolveCache, s::Integer, z::AbstractArray{T,3}, t::AbstractArray{T,3}) where {T}
	Hz = _h_apply(z, m.mpo[s], m.hstorage[s], m.hstorage[s+1])
	return real(dot(z, Hz)) - 2 * real(dot(z, t)) + m.ynorm2
end

"""
	leftsweep!(m::LinsolveCache, alg) -> kvals

One left-to-right ALS sweep: at each site the local normal equation H z = t is solved,
the chain is moved by QR and both environment stacks incremented. `kvals` collects the
exact global residual² ‖A·x − y‖² after every site update (non-increasing).
"""
function leftsweep!(m::LinsolveCache, alg::IterativeMPSAlgorithm)
	L = length(m.ket)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		z, t = _site_solve(m, s, _solverof(alg))
		kvals[s] = _site_loss(m, s, z, t)
		q, r = _gauge_left(z)
		m.ket[s] = q
		m.ket[s+1] = _contract_first(m.ket[s+1], r)
		_h_left!(m, s)
		_b_left!(m, s)
	end
	z, t = _site_solve(m, L, _solverof(alg))
	kvals[L] = _site_loss(m, L, z, t)
	m.ket[L] = z
	return kvals
end

"""
	rightsweep!(m::LinsolveCache, alg) -> kvals

One right-to-left ALS sweep (symmetric, LQ gauge moves). `kvals` is ordered by processing
time — sites `L, L-1, …, 1` — and collects the exact global residual² ‖A·x − y‖² after
every site update (non-increasing).
"""
function rightsweep!(m::LinsolveCache, alg::IterativeMPSAlgorithm)
	L = length(m.ket)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		z, t = _site_solve(m, s, _solverof(alg))
		kvals[k] = _site_loss(m, s, z, t)
		k += 1
		l, q = _gauge_right(z)
		m.ket[s] = q
		m.ket[s-1] = _contract_last(m.ket[s-1], l)
		_h_right!(m, s)
		_b_right!(m, s)
	end
	z, t = _site_solve(m, 1, _solverof(alg))
	kvals[L] = _site_loss(m, 1, z, t)
	m.ket[1] = z
	return kvals
end

sweep!(m::LinsolveCache, alg::IterativeMPSAlgorithm) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

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
					  Vector{Array{T,3}}(undef, length(y) + 1),
					  real(_dot(y, y)))
	_init_storages_right!(m)
	# the sweeps never touch Schmidt values: reset them so "initialized" always implies
	# "properly canonical"
	unset_svectors!(x)
	return m
end

"""
	linsolve!(x, A, y, alg::ALSLinSolve = ALSLinSolve()) -> x

Single-site variational (ALS) solve of `A·x ≈ y` refined in place on the initial guess
`x`; iterated by the generic `iterative_compute!` on the exact global residual². The
sweeps solve the raw-data problem, and the solution carries the per-site scale
`scaling(y)/scaling(A)` — the represented equation `A·x = y` holds with the operands'
external scales (a scaling^L power is never materialized).
"""
function linsolve!(x, A, y, alg::ALSLinSolve = ALSLinSolve())
	bonddim(x) != alg.D && changebond!(x; D=alg.D)
	m = LinsolveCache(A, y, x)
	iterative_compute!(m, alg)
	# the sweeps solve the raw-data problem A_data·x = y_data; the represented equation
	# (s_A^L·A_data)·(s_x^L·x) = s_y^L·y_data holds iff s_x = scaling(y)/scaling(A)
	setscaling!(m.ket, scaling(y) / _opscaling(A))
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
	linsolve(A, y, alg=ALSLinSolve()) -> x

Solve `A·x ≈ y` variationally: find a finite-bond MPS `x` of bond dimension `alg.D`
minimizing the residual norm `||A·x - y||` by single-site ALS sweeps (normal equation
A†A x = A† y). The initial guess is a random state. Block-sparse (Hamiltonian) inputs
are expanded to the dense MPO layer.
"""
function linsolve(A::AbstractMPO, y::CanonicalMPS, alg::ALSLinSolve = ALSLinSolve())
	_validate_linsolve(A, y)
	T = promote_type(scalartype(A), scalartype(y))
	x = randommps(T, ophydims(A); D=alg.D, normalize=false)
	return linsolve!(x, A, y, alg)
end

linsolve(A::MPOHamiltonian, y::CanonicalMPS, alg::ALSLinSolve = ALSLinSolve()) =
        linsolve(MPO(tompotensors(A)), y, alg)

# ---------- two-site (ALSLinSolve2) sweeps and interface ----------

# ⟨x|A†A|z⟩ over the pair: z2[zL, p1, p2, zR]; W1/W2 are the MPO site tensors (used
# as-is — the pair block is never materialized)
# hL legs (xL, A†-bond, A-bond, zL); hR legs (xR, A†-bond, A-bond, zR)
# contraction order: the large MPS bonds (zL, zR — far larger than the MPO bonds and
# physical dimensions) are folded into the environments first; every intermediate stays
# at O(x·z·D_Aᵏ·dᵏ) instead of materializing zL·zR-scaled blocks
function _h2_apply(z2::AbstractArray{T,4}, W1, W2,
				   hL::AbstractArray{T,4}, hR::AbstractArray{T,4}) where {T}
	# fold z2 into the right environment over the large ket bond zR
	t1 = @tensor t1[zL, i1, i2, xR, cR, w2] := hR[xR, cR, w2, zR] * z2[zL, i1, i2, zR]
	# apply the MPO block site by site (small physical and A-bond legs)
	t2 = @tensor t2[zL, i1, xR, cR, c, o2] := t1[zL, i1, i2, xR, cR, w2] *
											  W2[c, o2, w2, i2]
	t3 = @tensor t3[zL, xR, cR, w1, o1, o2] := t2[zL, i1, xR, cR, c, o2] *
											   W1[w1, o1, c, i1]
	# wrap with A†2: its input legs become the free physical legs of the target
	t4 = @tensor t4[zL, xR, w1, c, o1, p2] := conj(W2[c, o2, cR, p2]) *
											   t3[zL, xR, cR, w1, o1, o2]
	t5 = @tensor t5[zL, xR, w1, cL, p1, p2] := conj(W1[cL, o1, c, p1]) *
											   t4[zL, xR, w1, c, o1, p2]
	# close with the left environment over the large ket bond zL and the small A-bonds
	return @tensor y[xL, p1, p2, xR] := hL[xL, cL, w1, zL] * t5[zL, xR, w1, cL, p1, p2]
end

# ⟨x|A†|y⟩ over the pair: the two-site right-hand side. As in the single-site
# `_b_target`, y's physicals contract the MPO PO legs and the free legs are the MPO PI
# legs (A†y lives on A's input space). Large MPS bonds folded into the environments
# first (same order as `_h2_apply`).
function _b2_target(W1, W2, y2::AbstractArray{T,4}, bL::AbstractArray{T,3}, bR::AbstractArray{T,3}) where {T}
	t1 = @tensor t1[kL, o1, o2, xR, cR] := bR[xR, cR, kR] * y2[kL, o1, o2, kR]
	t2 = @tensor t2[kL, o1, xR, c, q2] := conj(W2[c, o2, cR, q2]) * t1[kL, o1, o2, xR, cR]
	t3 = @tensor t3[kL, xR, cL, q1, q2] := conj(W1[cL, o1, c, q1]) * t2[kL, o1, xR, c, q2]
	return @tensor t[xL, q1, q2, xR] := bL[xL, cL, kL] * t3[kL, xR, cL, q1, q2]
end

function _site_solve2(m::LinsolveCache, s::Integer, solver::KrylovKit.LinearSolver)
	W1, W2 = m.mpo[s], m.mpo[s+1]
	y2 = @tensor yy[a, p1, p2, b] := m.bra[s][a, p1, c] * m.bra[s+1][c, p2, b]
	t = _b2_target(W1, W2, y2, m.bstorage[s], m.bstorage[s+2])
	shape = (size(m.hstorage[s], 1), size(W1, 4), size(W2, 4), size(m.hstorage[s+2], 1))
	z0 = size(m.ket[s]) == shape ? m.ket[s] : zero(t)
	# the local normal equation is solved iteratively (KrylovKit) on the linear
	# operator action — the dense H matrix is never formed
	z, _ = KrylovKit.linsolve(y -> _h2_apply(y, W1, W2, m.hstorage[s], m.hstorage[s+2]),
							  t, z0, solver)
	return z, t
end

# the exact global residual² ‖A·x − y‖², evaluated from the local decomposition at the
# pair with the updated tensor z — no global contraction is needed
function _site_loss(m::LinsolveCache, s::Integer, z::AbstractArray{T,4}, t::AbstractArray{T,4}) where {T}
	Hz = _h2_apply(z, m.mpo[s], m.mpo[s+1], m.hstorage[s], m.hstorage[s+2])
	return real(dot(z, Hz)) - 2 * real(dot(z, t)) + m.ynorm2
end

function leftsweep!(m::LinsolveCache, alg::ALSLinSolve2)
	L = length(m.ket)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		z2, t = _site_solve2(m, s, _solverof(alg))
		kvals[s] = _site_loss(m, s, z2, t)
		_als2_update!(m.ket, s, z2, alg; move_right=true)
		_h_left!(m, s)
		_b_left!(m, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

function rightsweep!(m::LinsolveCache, alg::ALSLinSolve2)
	L = length(m.ket)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		z2, t = _site_solve2(m, s - 1, _solverof(alg))
		kvals[k] = _site_loss(m, s - 1, z2, t)
		k += 1
		_als2_update!(m.ket, s - 1, z2, alg; move_right=false)
		_h_right!(m, s)
		_b_right!(m, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(m::LinsolveCache, alg::ALSLinSolve2) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

function linsolve!(x, A, y, alg::ALSLinSolve2)
	m = LinsolveCache(A, y, x)
	iterative_compute!(m, alg)
	# the sweeps solve the raw-data problem A_data·x = y_data; the represented equation
	# (s_A^L·A_data)·(s_x^L·x) = s_y^L·y_data holds iff s_x = scaling(y)/scaling(A):
	# attach that per-site scale and fold the center norm into it (see `mult!`)
	setscaling!(m.ket, scaling(y) / _opscaling(A))
	_renormalize!(m.ket, m.ket[1], false)
	return x
end

function linsolve(A::AbstractMPO, y::CanonicalMPS, alg::ALSLinSolve2)
	_validate_linsolve(A, y)
	T = promote_type(scalartype(A), scalartype(y))
	x = randommps(T, ophydims(A); D=_guess_bond(alg.trunc), normalize=false)
	return linsolve!(x, A, y, alg)
end
linsolve(A::MPOHamiltonian, y::CanonicalMPS, alg::ALSLinSolve2) =
	linsolve(MPO(tompotensors(A)), y, alg)
