# DMRG1 ground-state search: unified leftsweep!/rightsweep!/sweep! interface over
# DMRGCache; the local problem is solved with KrylovKit.eigsolve

# ---------- DMRGCache: the ⟨ψ|W|ψ⟩ environment stack for DMRG1 / TDVP1 ----------

struct DMRGCache{M<:AbstractMPO, V<:CanonicalMPS, T}
	H::M
	ket::V
	hstorage::Vector{Array{T,3}}
end

"""
	DMRGCache(h::AbstractMPO, ψ::CanonicalMPS) -> DMRGCache

Build the environment stack `⟨ψ|W|ψ⟩` for a right-canonical guess `ψ`: `hstorage[1]` is
the left boundary, `hstorage[L+1]` the right boundary, and the right environments of all
sites are precomputed (matching the left-to-right first sweep of DMRG1 / TDVP1). One
full sweep is `leftsweep!` (sites `1:L`, QR moves the orthogonality center right)
followed by `rightsweep!` (sites `L:1`) — this order must not be exchanged.
"""
function DMRGCache(h::AbstractMPO, ψ::CanonicalMPS)
	L = length(ψ)
	T = promote_type(scalartype(h), scalartype(ψ))
	# the sweeps keep the mixed canonical form of the data but never touch the Schmidt
	# values: reset them on the guess, so "initialized" always implies "properly canonical"
	unset_svectors!(ψ)
	hs = Vector{Array{T,3}}(undef, L + 1)
	hs[1] = l_LL(ψ, h, ψ)
	hs[L+1] = r_RR(ψ, h, ψ)
	for s in L:-1:2
		hs[s] = _updateright(hs[s+1], ψ[s], h[s], ψ[s])
	end
	return DMRGCache(h, ψ, hs)
end

"""
	updateleft!(env, site)

Update the left environment of site `site+1` from the environment at `site`; the
orthogonality center moves to `site+1`.
"""
function updateleft!(env::DMRGCache, site::Integer)
	env.hstorage[site+1] = _updateleft(env.hstorage[site], env.ket[site], env.H[site], env.ket[site])
	return env
end

"""
	updateright!(env, site)

Update the right environment of site `site-1` from the environment at `site`; the
orthogonality center moves to `site-1`.
"""
function updateright!(env::DMRGCache, site::Integer)
	env.hstorage[site] = _updateright(env.hstorage[site+1], env.ket[site], env.H[site], env.ket[site])
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
		(alg.verbosity > 2) && _logupdate(stdout, "l2r", s, kvals[s])
		q, r = leftorth!(env.ket[s], (1, 2), (3,))
		env.ket[s] = q
		env.ket[s+1] = _contract_first(env.ket[s+1], r)
		updateleft!(env, s)
	end
	kvals[L], _ = _dmrg_local_update!(env, L, alg)
	(alg.verbosity > 2) && _logupdate(stdout, "l2r", L, kvals[L])
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
		(alg.verbosity > 2) && _logupdate(stdout, "r2l", s, kvals[k])
		k += 1
		l, q = rightorth!(env.ket[s], (1,), (2, 3))
		env.ket[s] = q
		env.ket[s-1] = _contract_last(env.ket[s-1], l)
		updateright!(env, s)
	end
	kvals[L], _ = _dmrg_local_update!(env, 1, alg)
	(alg.verbosity > 2) && _logupdate(stdout, "r2l", 1, kvals[L])
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
	bonddim(ψ) != alg.D && changebond!(ψ; D=alg.D)
	env = DMRGCache(h, ψ)
	khist = iterative_compute!(env, alg)
	setscaling!(ψ, 1.0)
	lmul!(1 / norm(ψ), ψ)   # data-normalized state (explosion-safe)
	return khist
end

"""
	ground_state(h::MPOHamiltonian, alg=DMRG1()) -> (E, ψ)

Ground-state energy and (canonical) state of the Hamiltonian `h`, starting from a
random initial state of bond dimension `alg.D`.
"""
function ground_state(h::MPOHamiltonian, alg::DMRG1=DMRG1())
	ψ = randommps(scalartype(h), ophydims(h); D=alg.D)
	khist = ground_state!(ψ, h, alg)
	(alg.verbosity > 1) && println("DMRG1 converged (delta = $(_iterative_delta(khist)))")
	return expectation(h, ψ), ψ
end
