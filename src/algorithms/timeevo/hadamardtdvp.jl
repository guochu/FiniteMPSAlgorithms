# HadamardTDVP: the time-dependent variational principle for the Hadamard (pointwise)
# product flow of two MPS chains,
#
#     dz/dτ = H ∘ z ,     (H ∘ z)(i₁…i_L) = H(i₁…i_L) · z(i₁…i_L) ,
#
# where the generator `H` is an MPS playing the role of a *diagonal* operator (a "field")
# and `z` the evolved state. One `sweep!(env, alg)` applies the elementwise exponential
#
#     z  ↦  exp(stepsize·H) .* z ,
#
# which is the flow TEMPO's `tdvpif` (src/influencefunctional/tdvpif/tdvpif.jl) uses to
# build the influence functional: there the generator is the influence operator in MPS
# form, the flow starts from the identity influence functional, and τ : 0 → 1 gives
# IF = e^H. The step-size convention is the one of the MPO TDVP (`TDVP1`/`TDVP2`):
# `stepsize` *is* the complex time increment, so `stepsize = -τ` cools by τ and
# `stepsize = -im*τ` evolves real time.
#
# This is single-site TDVP1 carried by the *three-chain* environments of the hadamard
# product ⟨z| H |z⟩ — the bra chain enters conjugated, the generator in the middle and
# the ket chain do not (the convention of `HadamardCache`) — with the local maps of the
# pointwise product: the physical index of `H[s]` is *shared* with the state's instead of
# being contracted. Environments and local targets are those of the ALS hadamard product
# (`_updateleft`, `_updateright`, `_reduce_hadamard_site`), and the complement-space
# (bond) maps coincide with `c_prime`, because the generator's bond passes straight
# through the bond.
#
# The state's bond profile *is* the manifold: like TDVP1, the sweeps neither grow nor
# truncate bonds. To let the flow populate the bonds — the point of the TDVPIF
# construction — zero-pad the guess up to the target profile first,
#
#     changebond!(ψ; D=D, noise=0)      # resize + rightorth! (SVD, NoTruncation)
#
# which leaves the padded directions orthonormal (so the sweeps can carry weight into
# them) and fixes the gauge; truncation is applied by the caller after the flow, e.g.
# `canonicalize!(ψ; alg=Orthogonalize(SVD(), trunc, false))`, as in `tdvpif`.

"""
	HadamardTDVP(; stepsize, ishermitian=false, verbosity=Defaults.verbosity)

Configuration of the Hadamard-product TDVP flow `dz/dτ = H ∘ z` between two MPS chains:
the generator `H` and the state `z` are both [`CanonicalMPS`](@ref), and their product is
the pointwise (Hadamard) one. `stepsize` is the complex time increment itself — one
`sweep!` applies the elementwise `exp(stepsize·H)`, so

- the cooling `z ↦ e^{-τ·H}.*z`: `stepsize = -τ`;
- the real-time flow `z ↦ e^{-i·τ·H}.*z`: `stepsize = -im*τ`.

`ishermitian` selects the Krylov exponentiate driver (Lanczos for hermitian, Arnoldi
otherwise) of the local projected generator. Only the bra side of the environments is
conjugated, so that generator is *not* hermitian in general — hence the `false` default,
as in TEMPO's `tdvpif` — and `true` is only safe for chains where the local maps are known
to be hermitian.

Drive it exactly like the MPO TDVP: build the cache once and repeat `sweep!(env, alg)`.

# Accuracy

On a manifold that can hold the pointwise product `H ∘ z` — in particular the full space
of a short chain, or a product state zero-padded to the profile the flow needs
(`changebond!(ψ; D=D, noise=0)`, which also canonicalizes the padded directions) — one
sweep reproduces the elementwise `exp(stepsize·H).*z` to roundoff, so several sweeps
accumulate the exact flow. On a restricted manifold the sweep realizes the projected flow
instead, which leaves a remainder linear in `stepsize` (halving `stepsize` halves it).
"""
@kwdef struct HadamardTDVP{S<:Number} <: MPSAlgorithm
	stepsize::S
	ishermitian::Bool = false
	verbosity::Int = Defaults.verbosity
end

"""
	HadamardTDVP2(; stepsize, trunc=DefaultTruncation, ishermitian=false, verbosity=Defaults.verbosity)

Configuration of the two-site Hadamard flow `dz/dτ = H ∘ z`. `stepsize` carries the same
convention as [`HadamardTDVP`](@ref) — the complex time increment itself, so one `sweep!`
applies the elementwise `exp(stepsize·H)` — and `trunc::TruncationScheme` caps the bond
dimensions of the pair SVD.

Where [`HadamardTDVP`](@ref) freezes the bond profile of the guess, `HadamardTDVP2` grows
it: the site pair is evolved against the two-site projected pointwise generator and
re-split with truncation, so a bond-dimension-1 guess reaches the profile the pointwise
product needs without any `changebond!` preparation, up to the cap carried by `trunc`. Use
it whenever the profile is unknown (the `tdvpif` preparation with `changebond!(ψ; D=D)` is
the single-site alternative), and `HadamardTDVP` when the profile is known and a cheaper
sweep is wanted.

The two conventions of the single-site variant hold unchanged: only the bra side of the
environments is conjugated, so the projected generator is generally not hermitian (Arnoldi
by default), and the generator's `scaling` is folded into the local generators as the
factor `scaling(H)^L`. The sweeps are documented with their implementations below.
"""
@kwdef struct HadamardTDVP2{S<:Number, TR<:TruncationScheme} <: MPSAlgorithm
	stepsize::S
	trunc::TR = DefaultTruncation
	ishermitian::Bool = false
	verbosity::Int = Defaults.verbosity
end

"""
	HadamardTDVPCache(H::CanonicalMPS, ψ::CanonicalMPS)

Environment stack of the Hadamard-product flow `dz/dτ = H ∘ z`: `state` (`ψ`) is the
evolved chain, `H` the diagonal generator, both MPS on the same lattice, and `hstorage`
the stack of the three-chain environments ⟨state| H |state⟩ at all bonds
(`hstorage[1]`/`hstorage[L+1]` the boundaries, the right environments precomputed). Each
environment carries legs `(bra bond, H bond, ket bond)`, exactly like the MPO environment
stack of a [`TDVPCache`](@ref) — the bra chain is the conjugated state, so the stack is
built from the state twice.

The state must be canonical (`iscanonical`) and must carry its bond profile already: the
sweeps are single-site and resize nothing, so padding the guess with
`changebond!(ψ; D=D, noise=0)` (which also right-canonicalizes with SVD/`NoTruncation`) is
the way to give the flow room to grow, as in TEMPO's `tdvpif`. The generator may be in any
gauge, and both chains may carry a non-unit `scaling` (the `scaling^L·∏tensors`
convention): the environments contract the site tensors, so the generator's `scaling` is
turned into the equivalent factor `scaling(H)^L` on the local generators (what
`tdvpif`'s `_absorb_scaling!(H)` achieves by absorbing it into the tensors), while the
state's `scaling` is bookkeeping only and flows through untouched.
"""
struct HadamardTDVPCache{A, B, T}
	H::A
	state::B
	hstorage::Vector{Array{T,3}}
end

"""
	HadamardTDVPCache(H::CanonicalMPS, ψ::CanonicalMPS) -> HadamardTDVPCache

Build the three-chain environment stack of the guess `ψ` for the Hadamard flow generated
by `H` (see [`HadamardTDVPCache`](@ref)); the Schmidt values of the guess are reset.
"""
function HadamardTDVPCache(H::CanonicalMPS, ψ::CanonicalMPS)
	L = length(ψ)
	(length(H) == L) ||
		throw(DimensionMismatch("generator and state must have the same length"))
	(phydims(H) == phydims(ψ)) ||
		throw(DimensionMismatch("physical dimensions of generator and state must match"))
	T = promote_type(scalartype(H), scalartype(ψ))
	unset_svectors!(ψ)
	hs = Vector{Array{T,3}}(undef, L + 1)
	hs[1] = ones(T, 1, 1, 1)
	hs[L + 1] = ones(T, 1, 1, 1)
	for s in L:-1:2
		hs[s] = _updateright(hs[s+1], ψ[s], H[s], ψ[s])
	end
	return HadamardTDVPCache(H, ψ, hs)
end

# ---------- environment transfers ----------
#
# The bra chain of the environments is the state itself (conjugated by `_updateleft`),
# so every transfer uses the state tensor on both the bra and the ket side, with the
# generator tensor in the middle — the `updatemultleft(h, AL, H, AL)` pattern of the
# TDVPIF sweeps.

function updateleft!(env::HadamardTDVPCache, site::Integer)
	env.hstorage[site+1] = _updateleft(env.hstorage[site], env.state[site], env.H[site],
									   env.state[site])
	return env
end

function updateright!(env::HadamardTDVPCache, site::Integer)
	env.hstorage[site] = _updateright(env.hstorage[site+1], env.state[site], env.H[site],
									  env.state[site])
	return env
end

# ---------- local generators ----------

# The flow must be driven by the generator's *represented value* (`value =
# scaling^L · ∏tensors`), but the environments contract the raw site tensors only — with a
# non-unit `scaling(H)` the sweeps would realize `exp(scaling(H)^-L·stepsize·H)`. TEMPO's
# `tdvpif` absorbs `scaling(H)` into the site tensors for exactly this reason; here the
# equivalent factor is put on the local generators instead, so the caller's chain is left
# alone. For the usual `scaling == 1` chains it is a no-op. The state's `scaling` is pure
# bookkeeping and does not enter the flow (its site tensors are isometric either way).
_hadamard_scale(env::HadamardTDVPCache) = scaling(env.H)^length(env.state)

# site generator: the tangent-space projection of `H[s] ∘ y`, i.e. the pointwise product
# of the generator and the state site tensors, their physical indices shared (a
# broadcast, not a contraction) — the local target of the ALS hadamard product
function _hadamard_site_map(env::HadamardTDVPCache, s::Int)
	c = _hadamard_scale(env)
	return y -> c .* _reduce_hadamard_site(env.H[s], y, env.hstorage[s], env.hstorage[s + 1])
end

# complement-space generators on the bond: the generator's bond passes straight through,
# both environments anchored at that bond (the right one folding in site `s+1`, the left
# one site `s-1`) — leg for leg the `c_prime` of the MPO case, since the environments
# carry the same (bra, middle, ket) legs
function _hadamard_bond_map_left(env::HadamardTDVPCache, s::Int)
	hs = env.hstorage
	c = _hadamard_scale(env)
	hright = _updateright(hs[s+2], env.state[s+1], env.H[s+1], env.state[s+1])
	return y -> c .* c_prime(y, hs[s+1], hright)
end

function _hadamard_bond_map_right(env::HadamardTDVPCache, s::Int)
	hs = env.hstorage
	c = _hadamard_scale(env)
	hleft = _updateleft(hs[s-1], env.state[s-1], env.H[s-1], env.state[s-1])
	return y -> c .* c_prime(y, hleft, hs[s])
end

# one Krylov exponential of a local projected generator
function _hadamard_exponentiate(f, t, x, alg::Union{HadamardTDVP,HadamardTDVP2})
	y, _ = exponentiate(f, t, x; ishermitian=alg.ishermitian,
						tol=Defaults.tol, krylovdim=25, maxiter=100)
	return y
end

# ---------- sweeps ----------

"""
	leftsweep!(env::HadamardTDVPCache, alg::HadamardTDVP) -> Float64[]

Left half of one Hadamard-flow step: every site tensor (including the last) evolves with
`exp(+dt/2·H_eff)` under the projected pointwise generator, every bond tensor with
`exp(-dt/2·H_bond)` (the generator's bond passing straight through), and the environments
follow the updated state — the same skeleton as the `TDVP1` left sweep.
"""
function _hadamard_leftsweep!(env::HadamardTDVPCache, alg::HadamardTDVP)
	st = env.state
	L = length(st)
	t = alg.stepsize / 2
	for s in 1:L-1
		st[s] = _hadamard_exponentiate(_hadamard_site_map(env, s), t, st[s], alg)
		st[s], v = _gauge_left(st[s])
		updateleft!(env, s)
		v = _hadamard_exponentiate(_hadamard_bond_map_left(env, s), -t, v, alg)
		st[s+1] = _contract_first(st[s+1], v)
	end
	st[L] = _hadamard_exponentiate(_hadamard_site_map(env, L), t, st[L], alg)
	return Float64[]
end

"""
	rightsweep!(env::HadamardTDVPCache, alg::HadamardTDVP) -> Float64[]

Right half of one Hadamard-flow step (symmetric to [`leftsweep!`](@ref)).
"""
function _hadamard_rightsweep!(env::HadamardTDVPCache, alg::HadamardTDVP)
	st = env.state
	L = length(st)
	t = alg.stepsize / 2
	for s in L:-1:2
		st[s] = _hadamard_exponentiate(_hadamard_site_map(env, s), t, st[s], alg)
		v, st[s] = _gauge_right(st[s])
		updateright!(env, s)
		v = _hadamard_exponentiate(_hadamard_bond_map_right(env, s), -t, v, alg)
		st[s-1] = _contract_last(st[s-1], v)
	end
	st[1] = _hadamard_exponentiate(_hadamard_site_map(env, 1), t, st[1], alg)
	return Float64[]
end

"""
	sweep!(env::HadamardTDVPCache, alg::HadamardTDVP)

One full Hadamard-product time step of size `alg.stepsize` (`leftsweep!` + `rightsweep!`),
i.e. the elementwise `env.state ↦ exp(stepsize·env.H) .* env.state`. Time-evolution loops
are driven by the caller: repeat `sweep!(env, alg)`.
"""
function sweep!(env::HadamardTDVPCache, alg::HadamardTDVP)
	_hadamard_leftsweep!(env, alg)
	_hadamard_rightsweep!(env, alg)
	return env
end

leftsweep!(env::HadamardTDVPCache, alg::HadamardTDVP) = _hadamard_leftsweep!(env, alg)
rightsweep!(env::HadamardTDVPCache, alg::HadamardTDVP) = _hadamard_rightsweep!(env, alg)

# ======================================================================================
# HadamardTDVP2: the two-site variant (configuration type defined next to HadamardTDVP)
# ======================================================================================
#
# Like `TDVP2` for the MPO generators, the pair of neighboring site tensors is evolved
# jointly against the projected *two-site* pointwise generator `H[s] ∘ H[s+1]` and
# re-split by a truncating SVD under `alg.trunc`. The bond dimensions are therefore
# adapted by the update itself — a product / bond-dimension-1 guess grows its bonds up to
# the cap `trunc` carries, with no `changebond!` padding — and are trimmed where the
# spectrum allows. The sweeps mirror `_tdvp2_leftsweep!`/`_tdvp2_rightsweep!`: pair step
# `exp(+dt/2·H_pair)`, truncating split with the singular values absorbed into the sweep
# direction, `updateleft!`/`updateright!`, and — except at the far edge of a sweep — the
# single-site *backward* half-step `exp(-dt/2·H_site)` on the fresh center (which is why
# the environments have to follow the state, exactly as in `TDVP2`).
#
# The pair target `_reduce_hadamard_site2` is the two-site counterpart of
# `_reduce_hadamard_site`: the generator's pair (its middle bond contracted) and the
# state's pair are fused with their physical indices shared, then contracted with the
# same three-chain environments at the bonds on both ends of the pair.

# two-site target: fuse the generator's pair with the state's pair (physical indices
# shared) and contract with the environments at the bonds on both ends
function _reduce_hadamard_site2(A1::MPSTensor, A2::MPSTensor, Θ::AbstractArray{T,4},
								cleft::AbstractArray{T,3}, cright::AbstractArray{T,3}) where {T}
	K12 = _two_site_tensor(A1, A2)              # [K_L, p1, p2, K_R]
	KB = reshape(K12, size(K12, 1), 1, size(K12, 2), size(K12, 3), size(K12, 4), 1) .*
		 reshape(Θ, 1, size(Θ, 1), size(Θ, 2), size(Θ, 3), 1, size(Θ, 4))
	@tensor mpsj[-1, -2, -3, -4] :=
		cleft[-1, 1, 2] * KB[1, 2, -2, -3, 3, 4] * cright[-4, 3, 4]
	return mpsj
end

function _hadamard_site2_map(env::HadamardTDVPCache, s::Int)
	hs = env.hstorage
	c = _hadamard_scale(env)
	return Θ -> c .* _reduce_hadamard_site2(env.H[s], env.H[s+1], Θ, hs[s], hs[s+2])
end

"""
	leftsweep!(env::HadamardTDVPCache, alg::HadamardTDVP2) -> Float64[]

Left half of one two-site Hadamard step: every pair of sites evolves with
`exp(+dt/2·H_pair)` and is re-split by a truncating SVD (singular values absorbed into the
right site of the pair, which moves the orthogonality center), followed — except for the
last pair — by the single-site backward half-step `exp(-dt/2·H_site)` on the new center.
The environments follow the updated state.
"""
function _hadamard2_leftsweep!(env::HadamardTDVPCache, alg::HadamardTDVP2)
	st = env.state
	L = length(st)
	t = alg.stepsize / 2
	for s in 1:L-1
		Θ = _two_site_tensor(st[s], st[s+1])
		Θ = _hadamard_exponentiate(_hadamard_site2_map(env, s), t, Θ, alg)
		st[s], st[s+1] = _split_two_site(Θ, alg.trunc; move_right=true)
		updateleft!(env, s)
		if s != L - 1
			st[s+1] = _hadamard_exponentiate(_hadamard_site_map(env, s + 1), -t, st[s+1], alg)
		end
	end
	return Float64[]
end

"""
	rightsweep!(env::HadamardTDVPCache, alg::HadamardTDVP2) -> Float64[]

Right half of one two-site Hadamard step (symmetric to [`leftsweep!`](@ref), singular
values absorbed into the left site of each pair).
"""
function _hadamard2_rightsweep!(env::HadamardTDVPCache, alg::HadamardTDVP2)
	st = env.state
	L = length(st)
	t = alg.stepsize / 2
	for s in L:-1:2
		Θ = _two_site_tensor(st[s-1], st[s])
		Θ = _hadamard_exponentiate(_hadamard_site2_map(env, s - 1), t, Θ, alg)
		st[s-1], st[s] = _split_two_site(Θ, alg.trunc; move_right=false)
		updateright!(env, s)
		if s != 2
			st[s-1] = _hadamard_exponentiate(_hadamard_site_map(env, s - 1), -t, st[s-1], alg)
		end
	end
	return Float64[]
end

"""
	sweep!(env::HadamardTDVPCache, alg::HadamardTDVP2)

One full two-site Hadamard time step of size `alg.stepsize` (`leftsweep!` + `rightsweep!`),
adapting the bond dimensions under `alg.trunc`.
"""
function sweep!(env::HadamardTDVPCache, alg::HadamardTDVP2)
	_hadamard2_leftsweep!(env, alg)
	_hadamard2_rightsweep!(env, alg)
	return env
end

leftsweep!(env::HadamardTDVPCache, alg::HadamardTDVP2) = _hadamard2_leftsweep!(env, alg)
rightsweep!(env::HadamardTDVPCache, alg::HadamardTDVP2) = _hadamard2_rightsweep!(env, alg)
