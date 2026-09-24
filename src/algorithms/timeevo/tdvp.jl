# TDVP1: single-site time-dependent variational principle, reusing the sweep interface.
# One `sweep!(env, alg)` advances the state by `alg.stepsize` (Strang splitting:
# left half-step + right half-step). `stepsize = -im*τ` gives imaginary-time evolution.
#
# Bond-tensor updates follow MPSKit's `C_hamiltonian`: both environments entering the
# bond effective Hamiltonian are anchored at the *same* MPO bond (the bond where the
# center tensor lives). The left half-step starts from the left edge (right environments
# precomputed by the cache), the right half-step from the right edge — the sweeps keep
# the mixed canonical form throughout.

"""
	TDVP1(; stepsize, D=Defaults.D, ishermitian=true, verbosity=Defaults.verbosity)

Configuration of single-site TDVP. `stepsize` is the time step (complex allowed;
`stepsize = -im*τ` evolves in imaginary time). `ishermitian` selects the KrylovKit
exponentiate driver (Lanczos for hermitian, Arnoldi otherwise).
"""
@kwdef struct TDVP1{S<:Number} <: MPSAlgorithm
	stepsize::S
	D::Int = Defaults.D
	ishermitian::Bool = true
	verbosity::Int = Defaults.verbosity
end

"""
	leftsweep!(env::DMRGCache, alg::TDVP1)

Left half of one TDVP1 time step: every site tensor (including the last) evolves with
`exp(+dt/2·H_eff)` and every bond tensor with `exp(-dt/2·H_bond)` (complement space,
both environments anchored at the bond).
"""
function leftsweep!(env::DMRGCache, alg::TDVP1)
	L = length(env.ket)
	dt = alg.stepsize
	t = -im * dt / 2
	for s in 1:L-1
		heff = Heff(env.H[s], env.hstorage[s], env.hstorage[s+1])
		x, _ = exponentiate(y -> ac_prime(y, heff), t, env.ket[s]; ishermitian=alg.ishermitian,
							tol=Defaults.tol, krylovdim=25, maxiter=100)
		env.ket[s] = x
		q, r = leftorth!(env.ket[s], (1, 2), (3,))
		env.ket[s] = q
		v = r
		updateleft!(env, s)
		# complement-space evolution on the bond: both environments anchored at bond s
		# (the right one folds in site s+1)
		hright = _updateright(env.hstorage[s+2], env.ket[s+1], env.H[s+1], env.ket[s+1])
		v, _ = exponentiate(y -> c_prime(y, env.hstorage[s+1], hright), -t, v;
							ishermitian=alg.ishermitian, tol=Defaults.tol, krylovdim=25, maxiter=100)
		env.ket[s+1] = _contract_first(env.ket[s+1], v)
	end
	heff = Heff(env.H[L], env.hstorage[L], env.hstorage[L+1])
	env.ket[L], _ = exponentiate(y -> ac_prime(y, heff), t, env.ket[L];
								 ishermitian=alg.ishermitian, tol=Defaults.tol, krylovdim=25, maxiter=100)
	return Float64[]
end

"""
	rightsweep!(env::DMRGCache, alg::TDVP1)

Right half of one TDVP1 time step (symmetric to `leftsweep!`).
"""
function rightsweep!(env::DMRGCache, alg::TDVP1)
	L = length(env.ket)
	dt = alg.stepsize
	t = -im * dt / 2
	for s in L:-1:2
		heff = Heff(env.H[s], env.hstorage[s], env.hstorage[s+1])
		x, _ = exponentiate(y -> ac_prime(y, heff), t, env.ket[s]; ishermitian=alg.ishermitian,
							tol=Defaults.tol, krylovdim=25, maxiter=100)
		env.ket[s] = x
		l, q = rightorth!(env.ket[s], (1,), (2, 3))
		env.ket[s] = q
		v = l
		updateright!(env, s)
		# complement-space evolution on the bond: both environments anchored at bond s-1
		# (the left one folds in site s-1)
		hleft = _updateleft(env.hstorage[s-1], env.ket[s-1], env.H[s-1], env.ket[s-1])
		v, _ = exponentiate(y -> c_prime(y, hleft, env.hstorage[s]), -t, v;
							ishermitian=alg.ishermitian, tol=Defaults.tol, krylovdim=25, maxiter=100)
		env.ket[s-1] = _contract_last(env.ket[s-1], v)
	end
	heff = Heff(env.H[1], env.hstorage[1], env.hstorage[2])
	env.ket[1], _ = exponentiate(y -> ac_prime(y, heff), t, env.ket[1];
								 ishermitian=alg.ishermitian, tol=Defaults.tol, krylovdim=25, maxiter=100)
	return Float64[]
end

"""
	sweep!(env::DMRGCache, alg::TDVP1)

One full TDVP1 time step of size `alg.stepsize` (`leftsweep!` + `rightsweep!`).
Time evolution loops are driven by the caller: repeat `sweep!(env, alg)`.
"""
sweep!(env::DMRGCache, alg::TDVP1) = begin
	leftsweep!(env, alg)
	rightsweep!(env, alg)
	return env
end

# ---------- MPO-TDVP1: TDVP on the MPO manifold (density-operator evolution) ----------
#
# The optimized state is a CanonicalMPO ρ (rank-4 site tensors [aL, po, aR, pin]); the
# generator is another MPO h. The flow is selected by `alg.stepsize`:
#
#   real dt:            ρ̇ = -i[h, ρ]              (unitary / Liouville evolution)
#   imaginary dt=-im·τ: ρ̇ = -(hρ + ρh)/2          (symmetric cooling: ρ(τ) = e^{-τh/2} ρ₀ e^{-τh/2},
#                                                 so evolving from ρ₀ ∝ identity by τ gives β_eff = τ/2)
#
# Both flows are linear in ρ with the two multiplier terms (hρ) and (ρh). The local
# effective maps follow from the Dirac-Frenkel projection with the Hilbert-Schmidt trace
# pairing tr((δρ)† h ρ): the three operator chains (δρ†, h, ρ) multiply SITE-WISE, the
# physical legs chain pairwise through each site product (conj(W).po ↔ h.po, h.pin ↔
# W.po, W.pin ↔ conj(W).pin), and the two multiplier terms keep two environment stacks
# over (bra bond, h bond, ρ bond) resp. (bra bond, ρ bond, h bond):
#
#   (hρ):  Ẇ[aL,po,aR,pin] = Σ E[aL,xl,wl]·h[xl,po,xr,m]·W[wl,m,wr,pin]·E[xr,wr]
#   (ρh):  Ẇ[aL,po,aR,pin] = Σ F[aL,wl,xl]·W[wl,po,wr,k]·h[xl,k,xr,pin]·F[aR,wr,xr]
#
# (E's legs: (bra, h bond, ρ bond); F's legs: (bra, ρ bond, h bond).) The bond
# (complement-space) maps act on the rank-2 gauge factor with the same two sandwiches
# (no physical legs to chain).

# term (hρ) site map
function _flow_site_left(W::MPOTensor, K::MPOTensor,
						 EL::AbstractArray{T,3}, ER::AbstractArray{T,3}) where {T}
	@tensor y[aL, po, aR, pin] := EL[aL, xl, wl] * K[xl, po, xr, m] * W[wl, m, wr, pin] * ER[aR, xr, wr]
	return y
end

# term (ρh) site map
function _flow_site_right(W::MPOTensor, K::MPOTensor,
						  FL::AbstractArray{T,3}, FR::AbstractArray{T,3}) where {T}
	@tensor y[aL, po, aR, pin] := FL[aL, wl, xl] * W[wl, po, wr, k] * K[xl, k, xr, pin] * FR[aR, wr, xr]
	return y
end

# bond (complement-space) generators on the rank-2 gauge factor
function _flow_bond_left(v::AbstractMatrix, EL::AbstractArray{T,3}, ER::AbstractArray{T,3}) where {T}
	@tensor y[l, r] := EL[l, x, w] * v[w, w2] * ER[r, x, w2]
	return y
end

function _flow_bond_right(v::AbstractMatrix, FL::AbstractArray{T,3}, FR::AbstractArray{T,3}) where {T}
	@tensor y[l, r] := FL[l, w, x] * v[w, w2] * FR[r, w2, x]
	return y
end

# real stepsizes give the commutator map (unitary flow), imaginary ones the
# anticommutator map (cooling); both maps are Hermitian w.r.t. the HS inner product
_flow_commutator(dt::Number) = isreal(dt)

function _flow_site(K::MPOTensor, EL::AbstractArray{T,3}, ER::AbstractArray{T,3},
					FL::AbstractArray{T,3}, FR::AbstractArray{T,3}, commutator::Bool) where {T}
	commutator ?
	(y -> _flow_site_left(y, K, EL, ER) - _flow_site_right(y, K, FL, FR)) :
	(y -> (_flow_site_left(y, K, EL, ER) + _flow_site_right(y, K, FL, FR)) / 2)
end

function _flow_bond(EL::AbstractArray{T,3}, ER::AbstractArray{T,3},
					FL::AbstractArray{T,3}, FR::AbstractArray{T,3}, commutator::Bool) where {T}
	commutator ?
	(v -> _flow_bond_left(v, EL, ER) - _flow_bond_right(v, FL, FR)) :
	(v -> (_flow_bond_left(v, EL, ER) + _flow_bond_right(v, FL, FR)) / 2)
end

"""
	MPOTDVPCache{M<:AbstractMPO, V<:CanonicalMPO, T}

Environment stack for the MPO-TDVP1: `H` the generator MPO, `rho` the evolved
CanonicalMPO, `estorage` the (hρ)-term environments `E[l, x, w]` over (bra bond, H
bond, ρ bond) and `estorageB` the (ρh)-term environments `F[l, w, x]` over (bra bond,
ρ bond, H bond) at all bonds (`estorage[1]`/`estorageB[1]` the left boundary,
`estorage[L+1]`/`estorageB[L+1]` the right boundary, right environments precomputed).
"""
struct MPOTDVPCache{M<:AbstractMPO, V<:CanonicalMPO, T}
	H::M
	rho::V
	estorage::Vector{Array{T,3}}
	estorageB::Vector{Array{T,3}}
end

MPOTDVPCache(h::MPOHamiltonian, ρ::CanonicalMPO) = MPOTDVPCache(MPO(h), ρ)

function MPOTDVPCache(h::MPO, ρ::CanonicalMPO)
	L = length(ρ)
	# the sweeps keep the mixed canonical form of the data but never touch the Schmidt
	# values: reset them on the guess, so "initialized" always implies "properly canonical"
	unset_svectors!(ρ)
	Tb = promote_type(scalartype(ρ), scalartype(h))
	es = Vector{Array{Tb,3}}(undef, L + 1)
	fs = Vector{Array{Tb,3}}(undef, L + 1)
	# the dense MPO's boundary bonds carry only the vacuum channel
	es[1] = ones(Tb, space_l(ρ), space_l(h), space_l(ρ))
	es[L+1] = ones(Tb, space_r(ρ), space_r(h), space_r(ρ))
	fs[1] = ones(Tb, space_l(ρ), space_l(ρ), space_l(h))
	fs[L+1] = ones(Tb, space_r(ρ), space_r(ρ), space_r(h))
	for s in L:-1:2
		es[s] = _flow_updateright(es[s+1], ρ[s], h[s])
		fs[s] = _flow_updateright_B(fs[s+1], ρ[s], h[s])
	end
	return MPOTDVPCache(h, ρ, es, fs)
end

function updateleft!(env::MPOTDVPCache, site::Integer)
	env.estorage[site+1] = _flow_updateleft(env.estorage[site], env.rho[site], env.H[site])
	env.estorageB[site+1] = _flow_updateleft_B(env.estorageB[site], env.rho[site], env.H[site])
	return env
end

function updateright!(env::MPOTDVPCache, site::Integer)
	env.estorage[site] = _flow_updateright(env.estorage[site+1], env.rho[site], env.H[site])
	env.estorageB[site] = _flow_updateright_B(env.estorageB[site+1], env.rho[site], env.H[site])
	return env
end

# environment transfers. Per site the three operator chains (conj(W), K, W) multiply
# with the physical legs chained pairwise (conj(W).po ↔ K.po, K.pin ↔ W.po, W.pin ↔
# conj(W).pin); the two stacks differ in which chain runs on the middle leg.

# (hρ) stack, legs (bra, h, ρ):
function _flow_updateleft(E::AbstractArray{T,3}, W::MPOTensor, K::MPOTensor) where {T}
	@tensor E2[l, x, w] := E[l0, x0, w0] * conj(W[l0, p, l, q]) * K[x0, p, x, m] * W[w0, m, w, q]
	return E2
end

function _flow_updateright(E::AbstractArray{T,3}, W::MPOTensor, K::MPOTensor) where {T}
	@tensor E2[l, x, w] := E[l0, x0, w0] * conj(W[l, p, l0, q]) * K[x, p, x0, m] * W[w, m, w0, q]
	return E2
end

# (ρh) stack, legs (bra, ρ, h):
function _flow_updateleft_B(E::AbstractArray{T,3}, W::MPOTensor, K::MPOTensor) where {T}
	@tensor E2[l, w, x] := E[l0, w0, x0] * conj(W[l0, p, l, q]) * W[w0, p, w, m] * K[x0, m, x, q]
	return E2
end

function _flow_updateright_B(E::AbstractArray{T,3}, W::MPOTensor, K::MPOTensor) where {T}
	@tensor E2[l, w, x] := E[l0, w0, x0] * conj(W[l, p, l0, q]) * W[w, p, w0, m] * K[x, m, x0, q]
	return E2
end

"""
	leftsweep!(env::MPOTDVPCache, alg::TDVP1)

Left half of one MPO-TDVP1 time step: every site tensor (including the last) evolves
with the local flow generator and the center is moved by the operator-space QR
(leg groups `(1, 2, 4) | (3,)`); the gauge factor evolves in the complement space.
"""
function leftsweep!(env::MPOTDVPCache, alg::TDVP1)
	L = length(env.rho)
	dt = alg.stepsize
	t = -im * dt / 2
	comm = _flow_commutator(dt)
	h, ρ = env.H, env.rho
	for s in 1:L-1
		f = _flow_site(h[s], env.estorage[s], env.estorage[s+1],
					   env.estorageB[s], env.estorageB[s+1], comm)
		x, _ = exponentiate(f, t, ρ[s]; ishermitian=alg.ishermitian,
							tol=Defaults.tol, krylovdim=25, maxiter=100)
		ρ[s] = x
		q, r = leftorth!(ρ[s], (1, 2, 4), (3,))
		ρ[s] = permute(q, (1, 2, 4, 3))
		v = r
		updateleft!(env, s)
		ER = _flow_updateright(env.estorage[s+2], ρ[s+1], h[s+1])
		FR = _flow_updateright_B(env.estorageB[s+2], ρ[s+1], h[s+1])
		g = _flow_bond(env.estorage[s+1], ER, env.estorageB[s+1], FR, comm)
		v, _ = exponentiate(g, -t, v; ishermitian=alg.ishermitian,
							tol=Defaults.tol, krylovdim=25, maxiter=100)
		ρ[s+1] = _contract_first(ρ[s+1], v)
	end
	f = _flow_site(h[L], env.estorage[L], env.estorage[L+1],
				   env.estorageB[L], env.estorageB[L+1], comm)
	ρ[L], _ = exponentiate(f, t, ρ[L]; ishermitian=alg.ishermitian,
						   tol=Defaults.tol, krylovdim=25, maxiter=100)
	return Float64[]
end

"""
	rightsweep!(env::MPOTDVPCache, alg::TDVP1)

Right half of one MPO-TDVP1 time step (symmetric to `leftsweep!`, LQ gauge moves with
leg groups `(1,) | (2, 3, 4)`).
"""
function rightsweep!(env::MPOTDVPCache, alg::TDVP1)
	L = length(env.rho)
	dt = alg.stepsize
	t = -im * dt / 2
	comm = _flow_commutator(dt)
	h, ρ = env.H, env.rho
	for s in L:-1:2
		f = _flow_site(h[s], env.estorage[s], env.estorage[s+1],
					   env.estorageB[s], env.estorageB[s+1], comm)
		x, _ = exponentiate(f, t, ρ[s]; ishermitian=alg.ishermitian,
							tol=Defaults.tol, krylovdim=25, maxiter=100)
		ρ[s] = x
		lc, q = rightorth!(ρ[s], (1,), (2, 3, 4))
		ρ[s] = q
		v = lc
		updateright!(env, s)
		EL = _flow_updateleft(env.estorage[s-1], ρ[s-1], h[s-1])
		FL = _flow_updateleft_B(env.estorageB[s-1], ρ[s-1], h[s-1])
		g = _flow_bond(EL, env.estorage[s], FL, env.estorageB[s], comm)
		v, _ = exponentiate(g, -t, v; ishermitian=alg.ishermitian,
							tol=Defaults.tol, krylovdim=25, maxiter=100)
		ρ[s-1] = _contract_last(ρ[s-1], v)
	end
	f = _flow_site(h[1], env.estorage[1], env.estorage[2],
				   env.estorageB[1], env.estorageB[2], comm)
	ρ[1], _ = exponentiate(f, t, ρ[1]; ishermitian=alg.ishermitian,
						   tol=Defaults.tol, krylovdim=25, maxiter=100)
	return Float64[]
end

"""
	sweep!(env::MPOTDVPCache, alg::TDVP1)

One full MPO-TDVP1 time step of size `alg.stepsize` (`leftsweep!` + `rightsweep!`).
Time evolution loops are driven by the caller: repeat `sweep!(env, alg)`.
"""
sweep!(env::MPOTDVPCache, alg::TDVP1) = begin
	leftsweep!(env, alg)
	rightsweep!(env, alg)
	return env
end
