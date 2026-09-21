# DMRG1 ground-state search: unified leftsweep!/rightsweep!/sweep! interface over
# DMRGCache; the local problem is solved with KrylovKit.eigsolve

# ---------- DMRGCache: the ⟨ψ|W|ψ⟩ environment stack for DMRG1 / TDVP1 ----------

struct DMRGCache{M<:AbstractMPO, V<:CanonicalMPS, T}
	H::M
	ket::V
	hstorage::Vector{Array{T,3}}
	center::Base.RefValue{Int}
end

"""
	DMRGCache(h::AbstractMPO, ψ::CanonicalMPS; center=1) -> DMRGCache

Build the environment stack `⟨ψ|W|ψ⟩`: `hstorage[s]` is the environment of sites `1:s-1`,
`hstorage[L+1]` the right boundary; the environments are exact on both sides of `center`.
With the default `center=1` the right environments of all sites are precomputed (matching
the left-to-right first sweep of DMRG1 / TDVP1).
"""
function DMRGCache(h::AbstractMPO, ψ::CanonicalMPS; center::Integer=1)
	L = length(ψ)
	T = promote_type(scalartype(h), scalartype(ψ))
	hs = Vector{Array{T,3}}(undef, L + 1)
	hs[1] = l_LL(ψ, h, ψ)
	hs[L+1] = r_RR(ψ, h, ψ)
	for s in 1:center-1
		hs[s+1] = _updateleft(hs[s], ψ[s], h[s], ψ[s])
	end
	for s in L:-1:center+1
		hs[s] = _updateright(hs[s+1], ψ[s], h[s], ψ[s])
	end
	return DMRGCache(h, ψ, hs, Ref(Int(center)))
end

"""
	updateleft!(env, site)

Update the left environment of site `site+1` from the environment at `site`; moves `center` to `site+1`.
"""
function updateleft!(env::DMRGCache, site::Integer)
	env.hstorage[site+1] = _updateleft(env.hstorage[site], env.ket[site], env.H[site], env.ket[site])
	env.center[] = site + 1
	return env
end

"""
	updateright!(env, site)

Update the right environment of site `site-1` from the environment at `site`; moves `center` to `site-1`.
"""
function updateright!(env::DMRGCache, site::Integer)
	env.hstorage[site] = _updateright(env.hstorage[site+1], env.ket[site], env.H[site], env.ket[site])
	env.center[] = site - 1
	return env
end

"""
	recalculate!(env, ψ, center)

Attach a new state `ψ` and rebuild all environments up to `center`.
"""
function recalculate!(env::DMRGCache, ψ::CanonicalMPS, center::Integer=length(ψ))
	if ψ !== env.ket
		copy!(env.ket, ψ)
	end
	fresh = DMRGCache(env.H, env.ket; center)
	copy!(env.hstorage, fresh.hstorage)
	env.center[] = Int(center)
	return env
end

"""
	increase_bond!(env; D)

Grow the bond dimensions of the cached state by zero padding (see `increase_bond!(::CanonicalMPS)`),
then rebuild the environments.
"""
function increase_bond!(env::DMRGCache; D::Int=Defaults.D)
	increase_bond!(env.ket; D)
	fresh = DMRGCache(env.H, env.ket; center=env.center[])
	copy!(env.hstorage, fresh.hstorage)
	return env
end

# ---------- DMRG1 sweeps ----------

"""
	leftsweep!(env::DMRGCache, alg::DMRG1) -> kvals

One left-to-right DMRG1 sweep: at each site the single-site effective Hamiltonian
(`CentralHeff`) is diagonalized with `KrylovKit.eigsolve` (`:SR`), the site tensor updated,
the center moved by QR, and the environment incremented. Returns the per-site local energies.
"""
function leftsweep!(env::DMRGCache, alg::DMRG1)
	L = length(env.ket)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		kvals[s], _ = _dmrg_local_update!(env, s, alg)
		q, r = leftorth!(env.ket[s], (1, 2), (3,))
		env.ket[s] = q
		env.ket[s+1] = _contract_first(env.ket[s+1], r)
		updateleft!(env, s)
	end
	kvals[L], _ = _dmrg_local_update!(env, L, alg)
	return kvals
end

"""
	rightsweep!(env::DMRGCache, alg::DMRG1) -> kvals

One right-to-left DMRG1 sweep (symmetric to `leftsweep!`, LQ gauge moves). `kvals` is
ordered by processing time — sites `L, L-1, …, 1`.
"""
function rightsweep!(env::DMRGCache, alg::DMRG1)
	L = length(env.ket)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		kvals[k], _ = _dmrg_local_update!(env, s, alg)
		k += 1
		l, q = rightorth!(env.ket[s], (1,), (2, 3))
		env.ket[s] = q
		env.ket[s-1] = _contract_last(env.ket[s-1], l)
		updateright!(env, s)
	end
	kvals[L], _ = _dmrg_local_update!(env, 1, alg)
	return kvals
end

"""
	sweep!(env::DMRGCache, alg::DMRG1) -> kvals

One full DMRG1 sweep: `leftsweep!` followed by `rightsweep!`.
"""
sweep!(env::DMRGCache, alg::DMRG1) = vcat(leftsweep!(env, alg), rightsweep!(env, alg))

function _dmrg_local_update!(env::DMRGCache, s::Integer, alg::DMRG1)
	heff = Heff(env.H[s], env.hstorage[s], env.hstorage[s+1])
	vals, vecs, info = eigsolve(x -> ac_prime(x, heff), env.ket[s], 1, :SR;
								ishermitian=true, tol=max(alg.tol, 1.0e-12),
								krylovdim=30, maxiter=200, eager=true)
	env.ket[s] = vecs[1]
	return real(vals[1]), info
end

"""
	ground_state!(ψ::CanonicalMPS, h::AbstractMPO, alg::DMRG1=DMRG1()) -> khist

Ground-state search starting from the initial state `ψ` (modified in place; e.g.
`ψ = randommps(ComplexF64, ophydims(h); D=D)`). Returns `khist`, the per-sweep
local-energy history (the convergence measure follows `_iterative_delta(khist)`).
The returned state is data-normalized: the external scale is reset through the
`scaling` field, never materialized as a scaling^L power.
"""
function ground_state!(ψ::CanonicalMPS, h::AbstractMPO, alg::DMRG1=DMRG1())
	env = DMRGCache(h, ψ)
	khist = iterative_compute!(env, alg)
	setscaling!(ψ, 1.0)
	lmul!(1 / norm(ψ), ψ)   # data-normalized state (explosion-safe)
	return khist
end

"""
	ground_state(h::MPOHamiltonian, alg=DMRG1(); D=Defaults.D) -> (E, ψ)

Ground-state energy and (canonical) state of the Hamiltonian `h`, starting from a
random initial state of bond dimension `D`.
"""
function ground_state(h::MPOHamiltonian, alg::DMRG1=DMRG1(); D::Int=Defaults.D)
	ψ = randommps(scalartype(h), ophydims(h); D)
	khist = ground_state!(ψ, h, alg)
	(alg.verbosity > 0) && println("DMRG1 converged (delta = $(_iterative_delta(khist)))")
	return expectation(h, ψ), ψ
end
