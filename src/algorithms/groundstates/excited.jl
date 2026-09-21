# excited states: DMRG1 with orthogonality projectors against previously found states;
# reuses the same leftsweep!/rightsweep!/sweep! interface (no separate algorithm type)

# ---------- ExcitedStateCache: DMRG1 environments with orthogonality projectors ----------

struct ExcitedStateCache{M<:AbstractMPO, V<:CanonicalMPS, P<:CanonicalMPS, T, OC<:OverlapCache}
	H::M
	ket::V
	projectors::Vector{P}
	hstorage::Vector{Array{T,3}}
	cstorages::Vector{OC}
	center::Base.RefValue{Int}
end

"""
	ExcitedStateCache(h::AbstractMPO, ψ::CanonicalMPS, projectors::Vector{CanonicalMPS})

Environment cache for the excited-state search of `ψ`: one overlap cache per projector.
"""
function ExcitedStateCache(h::AbstractMPO, ψ::CanonicalMPS, projectors::Vector{<:CanonicalMPS})
	env = DMRGCache(h, ψ)
	cs = [OverlapCache(ψ, p) for p in projectors]
	return ExcitedStateCache(h, ψ, projectors, env.hstorage, cs, Ref(env.center[]))
end

function updateleft!(env::ExcitedStateCache, site::Integer)
	env.hstorage[site+1] = _updateleft(env.hstorage[site], env.ket[site], env.H[site], env.ket[site])
	for c in env.cstorages
		updateleft!(c, site)
	end
	env.center[] = site + 1
	return env
end

function updateright!(env::ExcitedStateCache, site::Integer)
	env.hstorage[site] = _updateright(env.hstorage[site+1], env.ket[site], env.H[site], env.ket[site])
	for c in env.cstorages
		updateright!(c, site)
	end
	env.center[] = site - 1
	return env
end

# ---------- projected DMRG1 sweeps ----------

"""
	leftsweep!(env::ExcitedStateCache, alg::DMRG1) -> kvals

One left-to-right projected DMRG1 sweep: the local problem minimizes
`⟨x|H_eff|x⟩` under the constraint `x ⊥ ψ₀⁽ˡ⁾` for all projectors `ψ₀⁽ˡ⁾`,
implemented as the projected linear map `P·H_eff`.
"""
function leftsweep!(env::ExcitedStateCache, alg::DMRG1)
	L = length(env.ket)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		kvals[s], _ = _projected_local_update!(env, s, alg)
		q, r = leftorth!(env.ket[s], (1, 2), (3,))
		env.ket[s] = q
		env.ket[s+1] = _contract_first(env.ket[s+1], r)
		updateleft!(env, s)
	end
	kvals[L], _ = _projected_local_update!(env, L, alg)
	return kvals
end

"""
	rightsweep!(env::ExcitedStateCache, alg::DMRG1) -> kvals

One right-to-left projected DMRG1 sweep. `kvals` is ordered by processing time —
sites `L, L-1, …, 1`.
"""
function rightsweep!(env::ExcitedStateCache, alg::DMRG1)
	L = length(env.ket)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		kvals[k], _ = _projected_local_update!(env, s, alg)
		k += 1
		l, q = rightorth!(env.ket[s], (1,), (2, 3))
		env.ket[s] = q
		env.ket[s-1] = _contract_last(env.ket[s-1], l)
		updateright!(env, s)
	end
	kvals[L], _ = _projected_local_update!(env, 1, alg)
	return kvals
end

sweep!(env::ExcitedStateCache, alg::DMRG1) = vcat(leftsweep!(env, alg), rightsweep!(env, alg))

function _projected_local_update!(env::ExcitedStateCache, s::Integer, alg::DMRG1)
	heff = Heff(env.H[s], env.hstorage[s], env.hstorage[s+1])
	# effective projector tensors: bra-side environments of the projectors at site s
	peff = [@tensor pe[-1, -2, -3] := c.cstorage[s][-1, 1] * c.ket[s][1, -2, 2] * c.cstorage[s+1][-3, 2]
			for c in env.cstorages]
	f(x) = begin
		y = ac_prime(x, heff)
		for p in peff
			y = y .- dot(p, y) .* p
		end
		return y
	end
	vals, vecs, info = eigsolve(f, env.ket[s], 1, :SR;
								ishermitian=true, tol=max(alg.tol, 1.0e-12),
								krylovdim=30, maxiter=200, eager=true)
	env.ket[s] = vecs[1]
	return real(vals[1]), info
end

"""
	excited_state!(ψ::CanonicalMPS, h::AbstractMPO, projectors::Vector{CanonicalMPS},
				   alg::DMRG1=DMRG1()) -> khist

Search for the lowest state orthogonal to all `projectors`, starting from the initial
state `ψ` (modified in place; e.g. `ψ = randommps(ComplexF64, ophydims(h); D=D)`).
Returns `khist`, the per-sweep local-energy history.
"""
function excited_state!(ψ::CanonicalMPS, h::AbstractMPO, projectors::Vector{<:CanonicalMPS}, alg::DMRG1=DMRG1())
	bonddim(ψ) != alg.D && changebond!(ψ; D=alg.D)
	env = ExcitedStateCache(h, ψ, projectors)
	khist = iterative_compute!(env, alg)
	return khist
end

"""
	excited_state(h::MPOHamiltonian, alg::DMRG1, ψ0::CanonicalMPS...) -> (E, ψ)

The lowest excited state orthogonal to all given previous states `ψ0`, starting from a
random initial state of bond dimension `alg.D`.
"""
function excited_state(h::MPOHamiltonian, alg::DMRG1, ψ0::CanonicalMPS...)
	ψ = randommps(scalartype(h), ophydims(h); D=alg.D)
	khist = excited_state!(ψ, h, collect(ψ0), alg)
	(alg.verbosity > 0) && println("excited DMRG1 converged (delta = $(_iterative_delta(khist)))")
	return expectation(h, ψ), ψ
end

excited_state(h::MPOHamiltonian, ψ0::CanonicalMPS...) =
	excited_state(h, DMRG1(), ψ0...)
