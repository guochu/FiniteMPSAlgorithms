# TDVP1: single-site time-dependent variational principle, reusing the sweep interface.
# One `sweep!(env, alg)` advances the state by the complex time increment `alg.stepsize`
# (Strang splitting: left half-step + right half-step), applying `exp(stepsize·H)`:
#
#   stepsize = -im*τ  ->  real-time evolution by τ,   exp(-i·τ·H)
#   stepsize = -τ     ->  imaginary-time evolution by τ,  exp(-τ·H)
#
# The algorithm applies the *left* action of a generator MPO `h` to a canonical chain, and
# both chain kinds run through `TDVPCache` with one environment stack and the same sweeps:
#
#   state::CanonicalMPS  —  standard MPS TDVP1:              |ψ⟩ ↦ exp(stepsize·h)|ψ⟩
#   state::CanonicalMPO  —  density-operator cooling (h·ρ):    ρ  ↦ exp(stepsize·h)·ρ
#
# The density-operator form is meaningful for imaginary time only; real-time evolution of
# a density operator is two-sided (`exp(-i·t·h)·ρ·exp(+i·t·h)`) and belongs to a
# `DMRGCache` on the superoperator — see the `TDVPCache` docstring. `DMRGCache` (MPS
# state) runs the very same MPS sweeps.
#
# Bond-tensor updates follow MPSKit's `C_hamiltonian`: both environments entering the
# bond effective Hamiltonian are anchored at the *same* MPO bond (the bond where the
# center tensor lives). The left half-step starts from the left edge (right environments
# precomputed by the cache), the right half-step from the right edge — the sweeps keep
# the mixed canonical form throughout.

"""
	TDVP1(; stepsize, D=Defaults.D, ishermitian=true, verbosity=Defaults.verbosity)

Configuration of single-site TDVP. `stepsize` is the complex time increment itself: one
`sweep!` applies `exp(stepsize·H)` (as two half-steps of `exp(stepsize/2·H)`), so

- real-time evolution by `τ`: `stepsize = -im*τ`, i.e. `exp(-i·τ·H)`;
- imaginary-time evolution (cooling) by `τ`: `stepsize = -τ`, i.e. `exp(-τ·H)`.

`ishermitian` selects the KrylovKit exponentiate driver (Lanczos for hermitian, Arnoldi
otherwise).
"""
@kwdef struct TDVP1{S<:Number} <: MPSAlgorithm
	stepsize::S
	D::Int = Defaults.D
	ishermitian::Bool = true
	verbosity::Int = Defaults.verbosity
end

# ---------- TDVPCache: one environment stack for the left action of a generator MPO ----------

"""
	TDVPCache{M<:AbstractMPO, V<:Union{CanonicalMPO,CanonicalMPS}, T}

Environment stack for single-site TDVP1 under the **left** action of the generator MPO
`H`: `state` is the evolved canonical chain, `estorage` the stack of `(bra bond, H bond,
ket bond)` environments at all bonds (`estorage[1]` the left boundary, `estorage[L+1]`
the right boundary, the right environments precomputed). There is only one stack — the
generator acts from the left only — and it has the same shape as the `hstorage` of a
[`DMRGCache`](@ref). The type of `state` selects the manifold and the flow:

# Usage

`alg.stepsize` is the complex time increment itself, so one `sweep!` applies
`exp(stepsize·H)`: `stepsize = -τ` cools by `τ` (`exp(-τ·H)`), `stepsize = -im*τ` evolves
real time by `τ` (`exp(-i·τ·H)`).

- `state::CanonicalMPS` (plain `MPO`/`MPOHamiltonian` generator, matching physical
  dimensions) — the ordinary MPS TDVP1, for real and imaginary time alike. Environment
  stack and sweeps coincide with those of a `DMRGCache` holding the same `H` and the same
  state; `TDVPCache` merely adds the density-operator manifold below.

- `state::CanonicalMPO` (generator with matching operator dimensions) — the state evolves
  by the *left action* of the generator, i.e. one `sweep!` implements

      ρ ↦ exp(stepsize·H)·ρ,       so dρ/dτ = -H·ρ   for `stepsize = -τ`.

  This imaginary-time step is the meaningful use: it is a cooling route to the Gibbs
  state. From the infinite-temperature state `ρ₀ ∝ identity`
  (`infinite_temperature_state`), `n` steps of size `τ` give
  `exp(-n·τ·H) ∝ ρ(β_eff = n·τ)` — the same `β_eff` as the symmetric two-sided cooling
  `exp(-τ/2·H)·ρ₀·exp(-τ/2·H)` at half the cost per site update.

- Real-time evolution of a density operator is **not** the left multiplication
  `exp(-i·t·H)·ρ` (which is what a complex `stepsize = -im*τ` would produce, i.e.
  `dρ/dt = -i·H·ρ`) but the two-sided `exp(-i·t·H)·ρ·exp(+i·t·H)`, which this cache does not
  implement — it stores the left-action environment only. Vectorize the density operator
  and drive it with a `DMRGCache` on the superoperator (the same manifold as an MPS chain
  of physical dimension `d²`):

  ```julia
  # unitary / Liouville flow  dρ/dt = -i[H, ρ]
  env = DMRGCache(superoperator(H, :left) - superoperator(H, :right), vectorize(ρ))
  sweep!(env, TDVP1(stepsize=-im * τ))

  # symmetric two-sided cooling (imaginary time)
  env = DMRGCache((superoperator(H, :left) + superoperator(H, :right)) / 2, vectorize(ρ))
  sweep!(env, TDVP1(stepsize=-τ))
  ```
"""
struct TDVPCache{M<:AbstractMPO, V<:Union{CanonicalMPO,CanonicalMPS}, T}
	H::M
	state::V
	estorage::Vector{Array{T,3}}
end

# ---------- density-operator environments: the left action `H·ρ` ----------
#
# For a density operator ρ on the MPO manifold the generator acts on the LEFT only, so a
# single environment stack `E[l, x, w]` over (bra bond, H bond, ρ bond) suffices. The
# three operator chains (conj(ρ), H, ρ) multiply site-wise, the physical legs chain
# pairwise (conj(ρ).po ↔ H.po, H.pin ↔ ρ.po) and ρ's input leg closes on itself across
# bra and ket:
#
#   Ẇ[aL,po,aR,pin] = Σ E[aL,xl,wl]·H[xl,po,xr,m]·W[wl,m,wr,pin]·E[xr,wr]
#
# Its index structure is exactly that of the MPS environment `⟨ψ|H|ψ⟩` (`ac_prime`), up to
# the extra self-paired physical leg; the complement-space map is the shared `c_prime`.

# single-site left-multiplication map on an MPO site tensor
function _flow_site_left(W::MPOTensor, K::MPOTensor,
						 EL::AbstractArray{T,3}, ER::AbstractArray{T,3}) where {T}
	@tensor y[aL, po, aR, pin] := EL[aL, xl, wl] * K[xl, po, xr, m] * W[wl, m, wr, pin] * ER[aR, xr, wr]
	return y
end

# left-to-right environment transfer
function _flow_updateleft(E::AbstractArray{T,3}, W::MPOTensor, K::MPOTensor) where {T}
	@tensor E2[l, x, w] := E[l0, x0, w0] * conj(W[l0, p, l, q]) * K[x0, p, x, m] * W[w0, m, w, q]
	return E2
end

# right-to-left environment transfer
function _flow_updateright(E::AbstractArray{T,3}, W::MPOTensor, K::MPOTensor) where {T}
	@tensor E2[l, x, w] := E[l0, x0, w0] * conj(W[l, p, l0, q]) * K[x, p, x0, m] * W[w, m, w0, q]
	return E2
end

# ---------- constructors ----------

"""
	TDVPCache(h::AbstractMPO, state) -> TDVPCache

Build the environment stack of the right-canonical guess `state` for the left action of
`h`; the type of `state` selects the manifold (see [`TDVPCache`](@ref)). The constructor
resets the Schmidt values of the guess, so "initialized" always implies "properly
canonical". For a `CanonicalMPS` state, `h` is the MPO acting on the state directly (for
a vectorized density-operator flow, the superoperator of [`superoperator`](@ref)); for a
`CanonicalMPO` state it is the operator multiplying the density operator from the left.
"""
function TDVPCache(h::AbstractMPO, ψ::CanonicalMPS)
	L = length(ψ)
	T = promote_type(scalartype(h), scalartype(ψ))
	unset_svectors!(ψ)
	es = Vector{Array{T,3}}(undef, L + 1)
	es[1] = l_LL(ψ, h, ψ)
	es[L+1] = r_RR(ψ, h, ψ)
	for s in L:-1:2
		es[s] = _updateright(es[s+1], ψ[s], h[s], ψ[s])
	end
	return TDVPCache(h, ψ, es)
end

TDVPCache(h::MPOHamiltonian, ρ::CanonicalMPO) = TDVPCache(MPO(h), ρ)

function TDVPCache(h::MPO, ρ::CanonicalMPO)
	L = length(ρ)
	unset_svectors!(ρ)
	Tb = promote_type(scalartype(ρ), scalartype(h))
	es = Vector{Array{Tb,3}}(undef, L + 1)
	# the dense MPO's boundary bonds carry only the vacuum channel
	es[1] = ones(Tb, space_l(ρ), space_l(h), space_l(ρ))
	es[L+1] = ones(Tb, space_r(ρ), space_r(h), space_r(ρ))
	for s in L:-1:2
		es[s] = _flow_updateright(es[s+1], ρ[s], h[s])
	end
	return TDVPCache(h, ρ, es)
end

# ---------- the single-site sweeps, shared by both manifolds ----------
#
# The sweep skeleton is identical for an MPS and a density operator: evolve the center
# with `exp(±t·H_eff)`, move the orthogonality center by the manifold's gauge move, refresh
# the environment, and evolve the gauge factor in the complement space. Only the local
# generator map, the gauge groups (rank 3 vs rank 4 site tensors) and the environment
# transfer differ, and each of those dispatches on the state.

# the cached state and its (single) environment stack
_tdvp_state(env::DMRGCache) = env.ket
_tdvp_state(env::TDVPCache) = env.state
_tdvp_storage(env::DMRGCache) = env.hstorage
_tdvp_storage(env::TDVPCache) = env.estorage

# environment transfer across one site, for an MPS or a density-operator site tensor
_tdvp_env_left(E, A::MPSTensor, W) = _updateleft(E, A, W, A)
_tdvp_env_left(E, A::MPOTensor, W) = _flow_updateleft(E, A, W)
_tdvp_env_right(E, A::MPSTensor, W) = _updateright(E, A, W, A)
_tdvp_env_right(E, A::MPOTensor, W) = _flow_updateright(E, A, W)

function updateleft!(env::TDVPCache, site::Integer)
	env.estorage[site+1] = _tdvp_env_left(env.estorage[site], env.state[site], env.H[site])
	return env
end

function updateright!(env::TDVPCache, site::Integer)
	env.estorage[site] = _tdvp_env_right(env.estorage[site+1], env.state[site], env.H[site])
	return env
end

# local single-site generator `y ↦ H_eff·y`
function _tdvp_site_map(env::Union{DMRGCache,TDVPCache{<:Any,<:CanonicalMPS}}, s::Int)
	es = _tdvp_storage(env)
	return y -> ac_prime(y, Heff(env.H[s], es[s], es[s+1]))
end
function _tdvp_site_map(env::TDVPCache{<:Any,<:CanonicalMPO}, s::Int)
	return y -> _flow_site_left(y, env.H[s], env.estorage[s], env.estorage[s+1])
end

# complement-space generator on the bond of the left sweep: both environments anchored at
# bond `s`, the right one folding in site `s+1`
function _tdvp_bond_map_left(env, s::Int)
	es = _tdvp_storage(env)
	st = _tdvp_state(env)
	hright = _tdvp_env_right(es[s+2], st[s+1], env.H[s+1])
	return y -> c_prime(y, es[s+1], hright)
end

# ... and on the bond of the right sweep (bond `s-1`, the left one folding in site `s-1`)
function _tdvp_bond_map_right(env, s::Int)
	es = _tdvp_storage(env)
	st = _tdvp_state(env)
	hleft = _tdvp_env_left(es[s-1], st[s-1], env.H[s-1])
	return y -> c_prime(y, hleft, es[s])
end

# one Krylov exponential of a local generator
function _tdvp_exponentiate(f, t, x, alg::TDVP1)
	y, _ = exponentiate(f, t, x; ishermitian=alg.ishermitian,
						tol=Defaults.tol, krylovdim=25, maxiter=100)
	return y
end

"""
	leftsweep!(env, alg::TDVP1) -> Float64[]

Left half of one TDVP1 time step: every site tensor (including the last) evolves with
`exp(+dt/2·H_eff)` and every bond tensor with `exp(-dt/2·H_bond)` (complement space, both
environments anchored at the bond). The gauge moves are those of the manifold of
`env`'s state (MPS leg groups `(1, 2) | (3,)`, density-operator ones `(1, 2, 4) | (3,)`).
"""
function _tdvp_leftsweep!(env, alg::TDVP1)
	st = _tdvp_state(env)
	L = length(st)
	t = alg.stepsize / 2
	for s in 1:L-1
		st[s] = _tdvp_exponentiate(_tdvp_site_map(env, s), t, st[s], alg)
		st[s], v = _gauge_left(st[s])
		updateleft!(env, s)
		v = _tdvp_exponentiate(_tdvp_bond_map_left(env, s), -t, v, alg)
		st[s+1] = _contract_first(st[s+1], v)
	end
	st[L] = _tdvp_exponentiate(_tdvp_site_map(env, L), t, st[L], alg)
	return Float64[]
end

"""
	rightsweep!(env, alg::TDVP1) -> Float64[]

Right half of one TDVP1 time step (symmetric to `leftsweep!`, the gauge moves run
right-to-left).
"""
function _tdvp_rightsweep!(env, alg::TDVP1)
	st = _tdvp_state(env)
	L = length(st)
	t = alg.stepsize / 2
	for s in L:-1:2
		st[s] = _tdvp_exponentiate(_tdvp_site_map(env, s), t, st[s], alg)
		v, st[s] = _gauge_right(st[s])
		updateright!(env, s)
		v = _tdvp_exponentiate(_tdvp_bond_map_right(env, s), -t, v, alg)
		st[s-1] = _contract_last(st[s-1], v)
	end
	st[1] = _tdvp_exponentiate(_tdvp_site_map(env, 1), t, st[1], alg)
	return Float64[]
end

"""
	sweep!(env, alg::TDVP1)

One full TDVP1 time step of size `alg.stepsize` (`leftsweep!` + `rightsweep!`).
Time evolution loops are driven by the caller: repeat `sweep!(env, alg)`.
"""
function sweep!(env::Union{DMRGCache,TDVPCache}, alg::TDVP1)
	_tdvp_leftsweep!(env, alg)
	_tdvp_rightsweep!(env, alg)
	return env
end

leftsweep!(env::Union{DMRGCache,TDVPCache}, alg::TDVP1) = _tdvp_leftsweep!(env, alg)
rightsweep!(env::Union{DMRGCache,TDVPCache}, alg::TDVP1) = _tdvp_rightsweep!(env, alg)
