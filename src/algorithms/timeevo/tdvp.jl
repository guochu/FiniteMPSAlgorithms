# TDVP: the time-dependent variational principle on a canonical chain, in a single-site
# (`TDVP1`) and a two-site (`TDVP2`) variant, reusing the sweep interface. One
# `sweep!(env, alg)` advances the state by the complex time increment `alg.stepsize`
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

# Accuracy, and initial states of too low rank

The sweep is not invariant under a bond-gauge transformation of the state: it integrates
the same projected flow only while the guess carries a canonically determined gauge. If
the initial state is *rank deficient* on some bond — a product state whose bond profile
was grown with `changebond!(...; noise=0)`, which pads the new bond directions with exact
zeros, is the typical case — the canonicalization does not single out a gauge (the
completion of the null directions of the factorization is arbitrary) and the trajectory
becomes representation dependent: rotating one bond of the (bitwise unchanged) state can
shift a single sweep by `~1e-7`, and two equivalent representations of the same guess (the
density-operator and the vectorized routes) can drift apart by `~1e-2` in the represented
density operator. A full-rank guess has a unique gauge and the results agree to roundoff.
Padding with a small `noise` (the `changebond!` default of `1e-10`) removes the
degeneracy and restores gauge independence; recipes that must resize exactly (e.g.
thermal-state preparation, where noise would be amplified by the cooling flow) should be
aware of this limitation.

The guess must also be canonical at all (`iscanonical`), since the environments are those
of an isometric chain; `vectorize(infinite_temperature_state(...))` is not (its site
tensors are the plain identities) and needs a `rightorth!` (SVD, `NoTruncation`) first.
"""
@kwdef struct TDVP1{S<:Number} <: MPSAlgorithm
	stepsize::S
	D::Int = Defaults.D
	ishermitian::Bool = true
	verbosity::Int = Defaults.verbosity
end

"""
	TDVP2(; stepsize, trunc=DefaultTruncation, ishermitian=true, verbosity=Defaults.verbosity)

Configuration of two-site TDVP. `stepsize` carries the same convention as
[`TDVP1`](@ref) — the complex time increment itself, `-τ` for cooling by `τ` and `-im*τ`
for real time — and one `sweep!` advances the state by it.

Where `TDVP1` freezes the bond dimensions of the guess, `TDVP2` adapts them: the site pair
is evolved jointly against the two-site effective generator and re-split by a *truncating*
SVD under `trunc::TruncationScheme`, so a low-rank guess (a product state, or the
infinite-temperature state of [`infinite_temperature_state`](@ref)) grows its bonds as the
entanglement builds up, up to the cap `trunc` carries (`_truncation_bond`), and is trimmed
where the spectrum allows. The bond profile needs no preparation: no `changebond!` call is
required on the initial state.

The sweeps follow MPSKit's `TDVP2`: pair step `exp(+dt/2·H_pair)`, splitting SVD with the
singular values absorbed into the sweep direction (which is what moves the orthogonality
center), then — except at the far edge of each sweep — a single-site backward half-step
`exp(-dt/2·H_site)` on the new center. The state and generator conventions are those of the
cache (see [`TDVPCache`](@ref)): with a `CanonicalMPS` state this is ordinary two-site MPS
TDVP, with a `CanonicalMPO` state the density-operator left action

    ρ ↦ exp(stepsize·H)·ρ.

The local generators are evaluated against the environments of the *current* state — the
sweeps refresh them as the orthogonality center moves, which is also what MPSKit's lazily
recalculated environments amount to. They cannot be held fixed for the whole step: a pair
update may change the bond dimension, and the single-site backward step then needs the
environment of the *new* bond.

# The guess

The guess must be in canonical form (`iscanonical`), as for every algorithm driven by
`DMRGCache`/`TDVPCache`: the environments and the local generators are those of an
isometric chain, and only in that gauge do the pair and the backward single-site
projectors compose into the tangent-space flow. `vectorize(infinite_temperature_state(...))`
represents the identity exactly, but its site tensors are the plain identities and are
therefore *not* isometric — regauge it with `rightorth!` (SVD, `NoTruncation`), which
resizes nothing. The bond profile itself needs no preparation: unlike
`changebond!`-padded recipes, a bond-dimension-1 guess is grown by the pair updates
themselves, up to the cap `trunc` carries.

# Accuracy

Where the manifold is complete — a guess whose bonds already fill the profile the
truncation allows — the pair and backward projectors are all the identity and one sweep
reproduces `exp(stepsize·H)` to roundoff. While the bonds are still growing the sweep runs
on a restricted manifold and leaves a remainder linear in `stepsize`: on the
next-nearest-neighbour test model at L = 4 a β = 1 cooling comes out at 2.3e-3, 1.1e-3 and
5.7e-4 (density-matrix error) for 20, 40 and 80 sweeps, the same numbers MPSKit's `TDVP2`
produces. Once the profile has grown, only the truncation under `trunc` remains.
"""
@kwdef struct TDVP2{S<:Number, TR<:TruncationScheme} <: MPSAlgorithm
	stepsize::S
	trunc::TR = DefaultTruncation
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

# one Krylov exponential of a local generator (TDVP1 single-site or TDVP2 pair)
function _tdvp_exponentiate(f, t, x, alg::Union{TDVP1,TDVP2})
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

# ======================================================================================
# TDVP2: two-site time-dependent variational principle (configuration type above)
# ======================================================================================
#
# Unlike TDVP1 — whose tangent space is the single-site one, so a low-rank guess can never
# grow — TDVP2 updates a *neighboring pair* of site tensors jointly against the two-site
# effective generator and re-splits the result by an SVD under `alg.trunc`. The split
# absorbs the singular values into the sweep direction (moving the orthogonality center)
# and *adapts the bond dimension*: it grows out of a product / infinite-temperature guess
# up to the cap carried by `trunc`, and shrinks where the spectrum allows.
#
# The sweep mirrors MPSKit's `TDVP2`: for every pair, `exp(+dt/2·H_pair)` on the two-site
# tensor, then the truncating SVD, then — unless the pair sits at the far edge — the
# single-site *backward* half-step `exp(-dt/2·H_site)` on the fresh center (the
# complement-space evolution of the standard scheme). The environments are kept consistent
# with the evolving state instead of being frozen for the whole step, so no cache refresh
# is needed between the halves.
#
# The pair tensor of the density-operator manifold carries six legs,
# `Θ[aL, po1, po2, aR, pin1, pin2]`, and is grouped as `(aL, po1, pin1) | (po2, aR, pin2)`
# — the same split, leg for leg, as the `(aL, ph1) | (ph2, aR)` one of its vectorized
# image, the fused physical index of the vectorized chain counting `po` fastest.

# ---------- two-site tensors and their truncating SVD split ----------

# the pair of neighboring site tensors, contracted into one tensor
function _two_site_tensor(A::MPSTensor, B::MPSTensor)
	@tensor Θ[aL, p1, p2, aR] := A[aL, p1, b] * B[b, p2, aR]
	return Θ
end
function _two_site_tensor(A::MPOTensor, B::MPOTensor)
	@tensor Θ[aL, po1, po2, aR, pin1, pin2] := A[aL, po1, b, pin1] * B[b, po2, aR, pin2]
	return Θ
end

# Split the pair tensor, discarding under `trunc`. `move_right = true` (left sweep) leaves
# site s left-isometric and absorbs the singular values into site s+1, `move_right = false`
# (right sweep) does the mirror image — the same convention as the DMRG2 local update.
function _split_two_site(Θ::AbstractArray{T,4}, trunc::TruncationScheme; move_right::Bool) where {T}
	u, sv, v, _ = tsvd!(Θ, (1, 2), (3, 4); trunc=trunc)
	sm = Diagonal(sv)
	if move_right
		@tensor vnew[nb, p2, aR] := sm[nb, j] * v[j, p2, aR]
		return u, vnew
	else
		@tensor unew[aL, p1, nb] := u[aL, p1, j] * sm[j, nb]
		return unew, v
	end
end

# density-operator pair: legs (aL, po1, po2, aR, pin1, pin2), grouped (1,2,5) | (3,4,6)
function _split_two_site(Θ::AbstractArray{T,6}, trunc::TruncationScheme; move_right::Bool) where {T}
	u, sv, v, _ = tsvd!(Θ, (1, 2, 5), (3, 4, 6); trunc=trunc)
	up = permute(u, (1, 2, 4, 3))               # (aL, po1, r, pin1)
	sm = Diagonal(sv)
	if move_right
		@tensor vnew[nb, po2, aR, pin2] := sm[nb, j] * v[j, po2, aR, pin2]
		return up, vnew
	else
		@tensor unew[aL, po1, nb, pin1] := up[aL, po1, j, pin1] * sm[j, nb]
		return unew, v
	end
end

# ---------- two-site effective generators ----------
#
# The environments are those of the *current* state: a pair update may change the bond
# dimension (that is the point of TDVP2), so the environments entering the next local
# generator — and in particular the left environment of the single-site backward step on
# the fresh center — have to be rebuilt from the updated tensors. The sweeps therefore
# refresh them as they go (`updateleft!`/`updateright!`), exactly as MPSKit's lazily
# recalculated environments do.

# MPS: the two-site effective Hamiltonian of the DMRG2 engine
function _tdvp_site2_map(env::Union{DMRGCache,TDVPCache{<:Any,<:CanonicalMPS}}, s::Int)
	es = _tdvp_storage(env)
	return y -> ac2_prime(y, TwoSiteHeff(env.H[s], env.H[s+1], es[s], es[s+2]))
end

# density operator: the two-site left action `H·ρ`, the pair version of `_flow_site_left`
function _tdvp_site2_map(env::TDVPCache{<:Any,<:CanonicalMPO}, s::Int)
	es = env.estorage
	return y -> _flow_site2_left(y, env.H[s], env.H[s+1], es[s], es[s+2])
end

function _flow_site2_left(Θ::AbstractArray{T,6}, K1::MPOTensor, K2::MPOTensor,
						  EL::AbstractArray{T,3}, ER::AbstractArray{T,3}) where {T}
	# Θ[bl, m1, m2, br, pin1, pin2] holds the state's pair; the state's output physical legs
	# (its second and third) pair with the generator's input legs, its input legs are free
	@tensor y[aL, po1, po2, aR, pin1, pin2] :=
		EL[aL, xl, bl] * K1[xl, po1, x, m1] * K2[x, po2, xr, m2] *
		Θ[bl, m1, m2, br, pin1, pin2] * ER[aR, xr, br]
	return y
end

# ---------- sweeps ----------

"""
	leftsweep!(env, alg::TDVP2)

Left half of one TDVP2 step: every pair of sites evolves with `exp(+dt/2·H_pair)` and is
re-split by a truncating SVD (singular values absorbed into the right site of the pair),
followed — except for the last pair — by the single-site backward half-step
`exp(-dt/2·H_site)` on the new center. The environments follow the updated state.
"""
function _tdvp2_leftsweep!(env, alg::TDVP2)
	st = _tdvp_state(env)
	L = length(st)
	t = alg.stepsize / 2
	for s in 1:L-1
		Θ = _two_site_tensor(st[s], st[s+1])
		Θ = _tdvp_exponentiate(_tdvp_site2_map(env, s), t, Θ, alg)
		st[s], st[s+1] = _split_two_site(Θ, alg.trunc; move_right=true)
		updateleft!(env, s)
		if s != L - 1
			st[s+1] = _tdvp_exponentiate(_tdvp_site_map(env, s + 1), -t, st[s+1], alg)
		end
	end
	return Float64[]
end

"""
	rightsweep!(env, alg::TDVP2)

Right half of one TDVP2 step (symmetric to `leftsweep!`, singular values absorbed into the
left site of each pair).
"""
function _tdvp2_rightsweep!(env, alg::TDVP2)
	st = _tdvp_state(env)
	L = length(st)
	t = alg.stepsize / 2
	for s in L:-1:2
		Θ = _two_site_tensor(st[s-1], st[s])
		Θ = _tdvp_exponentiate(_tdvp_site2_map(env, s - 1), t, Θ, alg)
		st[s-1], st[s] = _split_two_site(Θ, alg.trunc; move_right=false)
		updateright!(env, s)
		if s != 2
			st[s-1] = _tdvp_exponentiate(_tdvp_site_map(env, s - 1), -t, st[s-1], alg)
		end
	end
	return Float64[]
end

"""
	sweep!(env, alg::TDVP2)

One full TDVP2 time step of size `alg.stepsize` (`leftsweep!` + `rightsweep!`).
"""
function sweep!(env::Union{DMRGCache,TDVPCache}, alg::TDVP2)
	_tdvp2_leftsweep!(env, alg)
	_tdvp2_rightsweep!(env, alg)
	return env
end

leftsweep!(env::Union{DMRGCache,TDVPCache}, alg::TDVP2) = _tdvp2_leftsweep!(env, alg)
rightsweep!(env::Union{DMRGCache,TDVPCache}, alg::TDVP2) = _tdvp2_rightsweep!(env, alg)
