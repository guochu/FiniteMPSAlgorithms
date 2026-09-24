# add: the unique exported interface for (truncating) chain summation
#   add(ψs, alg)  -> CanonicalMPS   (states)
#   add(ρs, alg)  -> CanonicalMPO   (density matrices)
# dispatched on alg: SVDCompression (exact block-diagonal sum + SVD sweep) or DMRG1 (ALS)

"""
	svd_add(ψs; trunc) -> (chain, err)

Greedy summation: exact block-diagonal sums folded pairwise, then a truncating
canonicalization at the end.
"""
function svd_add(ψs::Vector{<:CanonicalMPS}, trunc::TruncationScheme)
	isempty(ψs) && throw(ArgumentError("empty input"))
	acc = ψs[1]
	for i in 2:length(ψs)
		acc = acc + ψs[i]
	end
	return _canonicalize!(acc; alg=Orthogonalize(SVD(), trunc, false))
end
function svd_add(ρs::Vector{<:CanonicalMPO}, trunc::TruncationScheme)
	isempty(ρs) && throw(ArgumentError("empty input"))
	acc = ρs[1]
	for i in 2:length(ρs)
		acc = acc + ρs[i]
	end
	out = CanonicalMPO(acc.data)
	_, e = _canonicalize!(out; alg=Orthogonalize(SVD(), trunc, false))
	return out, e
end

# ALS cache: find bra maximizing Σ_n |⟨bra|kets[n]⟩|
struct AddCache{O, I, OC<:OverlapCache}
	bra::O
	kets::Vector{I}
	hstorage::Vector{OC}
end

"""
	AddCache(bra, kets)

Build the `AddCache` of the iterative `add`: every per-input overlap environment
(`OverlapCache`) is constructed and initialized here.
"""
function AddCache(bra, kets)
	unset_svectors!(bra)   # bra is optimized in place; sweeps never touch Schmidt values
	return AddCache(bra, kets, [OverlapCache(bra, ψ) for ψ in kets])
end

function _init_hstorage_right!(m::AddCache)
	L = length(m.bra)
	for c in m.hstorage
		T = scalartype(c.bra)
		c.cstorage[1] = ones(T, 1, 1)
		c.cstorage[L+1] = ones(T, 1, 1)
		for s in L:-1:2
			c.cstorage[s] = _updateright(c.cstorage[s+1], m.bra[s], c.ket[s])
		end
	end
	return m
end

function _reduce_site(m::AddCache, s::Integer)
	acc = missing
	for c in m.hstorage
		if c.ket[s] isa MPOTensor
			tj = @tensor tmp[-1, -2, -3, -4] := c.cstorage[s][-1, 1] * c.ket[s][1, -2, 2, -4] * c.cstorage[s+1][-3, 2]
		else
			tj = @tensor tmp[-1, -2, -3] := c.cstorage[s][-1, 1] * c.ket[s][1, -2, 2] * c.cstorage[s+1][-3, 2]
		end
		acc = acc isa Missing ? tj : acc + tj
	end
	acc isa Missing && throw(ArgumentError("empty AddCache"))
	return acc
end

function leftsweep!(m::AddCache, alg::DMRG1)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		mpsj = _reduce_site(m, s)
		kvals[s] = norm(mpsj)
		(alg.verbosity > 2) && _logupdate(stdout, "l2r", s, kvals[s])
		q, r = _gauge_left(mpsj)
		# normalize the gauge factor: ALS determines only the direction, and keeping the
		# bra data O(1) stabilizes the sweeps when inputs have tiny norms. The exact scale
		# of the sum is restored once at the end of `iterative_add`.
		rn = norm(r)
		rn == 0 || (r /= rn)
		m.bra[s] = q
		m.bra[s+1] = _contract_first(m.bra[s+1], r)
		for c in m.hstorage
			updateleft!(c, s)
		end
	end
	mpsj = _reduce_site(m, L)
	kvals[L] = norm(mpsj)
	(alg.verbosity > 2) && _logupdate(stdout, "l2r", L, kvals[L])
	m.bra[L] = mpsj
	return kvals
end

function rightsweep!(m::AddCache, alg::DMRG1)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		mpsj = _reduce_site(m, s)
		kvals[k] = norm(mpsj)
		(alg.verbosity > 2) && _logupdate(stdout, "r2l", s, kvals[k])
		k += 1
		l, q = _gauge_right(mpsj)
		ln = norm(l)
		ln == 0 || (l /= ln)
		m.bra[s] = q
		m.bra[s-1] = _contract_last(m.bra[s-1], l)
		for c in m.hstorage
			updateright!(c, s)
		end
	end
	mpsj = _reduce_site(m, 1)
	kvals[L] = norm(mpsj)
	(alg.verbosity > 2) && _logupdate(stdout, "r2l", 1, kvals[L])
	m.bra[1] = mpsj
	return kvals
end

sweep!(m::AddCache, alg::DMRG1) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

# ---------- initial guess & cache construction ----------

"""
	svdguess_add(chains, D::Int)

On-the-fly naive SVD of the exact block-diagonal sum as an initial guess for the
iterative `add` (the exact sum itself is never materialized): site i is the bond-wise
cat of the inputs, each input's external scaling is folded relative to the first input
into every site tensor (per site — a scaling^L power is never formed). The bond
dimension is capped at `D` (`truncdim(D)`).
"""
function svdguess_add(chains, D::Int)
	L = length(chains[1])
	s1 = scaling(chains[1])
	scales = [scaling(ψ) / s1 for ψ in chains]
	site = i -> begin
		if i == 1
			cat([scales[n] * chains[n][1] for n in eachindex(chains)]...; dims=3)
		elseif i == L
			cat([scales[n] * chains[n][L] for n in eachindex(chains)]...; dims=1)
		else
			cat([scales[n] * chains[n][i] for n in eachindex(chains)]...; dims=(1, 3))
		end
	end
	data, _ = _naive_svd_guess(site, L; trunc=truncdim(D))
	return chains[1] isa CanonicalMPS ?
		CanonicalMPS(Vector{Array{scalartype(data[1]),3}}(data)) :
		CanonicalMPO(Vector{Array{scalartype(data[1]),4}}(data))
end

"""
	add!(out, chains, alg::DMRG1) -> out

Single-site variational (ALS) sum of `chains` refined in place on the initial guess
`out`. The sweeps determine direction and magnitude at the data level; the sum is
expressed in the first input's per-site `scaling` convention (a scaling^L power is
never materialized).
"""
function add!(out, chains, alg::DMRG1)
	bonddim(out) != alg.D && changebond!(out; D=alg.D)
	cache = AddCache(out, chains)
	iterative_compute!(cache, alg)
	setscaling!(out, scaling(chains[1]))
	return out
end

"""
	add(ψs::Vector{<:CanonicalMPS}, alg=SVDCompression(trunc=DefaultTruncation)) -> CanonicalMPS
	add(ψs::Vector{<:CanonicalMPS}, alg::DMRG1) -> CanonicalMPS
	add(ρs::Vector{<:CanonicalMPO}, alg) / add(ρs, alg::DMRG1) -> CanonicalMPO

Sum of chains: the SVD route forms the exact block-diagonal sum and compresses it with
a single SVD sweep; the variational (ALS) route draws a `svdguess_add(ψs, alg.D)` initial
guess and refines it by sweeps. The exact sum is `Base.:+` (block-diagonal, no
truncation).
"""
function add(ψs::Vector{<:CanonicalMPS}, alg::SVDCompression=DefaultMultAlg)
	isempty(ψs) && throw(ArgumentError("empty input"))
	return svd_add(ψs, alg.trunc)[1]
end
function add(ψs::Vector{<:CanonicalMPS}, alg::DMRG1)
	isempty(ψs) && throw(ArgumentError("empty input"))
	return add!(svdguess_add(ψs, alg.D), ψs, alg)
end
function add(ρs::Vector{<:CanonicalMPO}, alg::SVDCompression=DefaultMultAlg)
	isempty(ρs) && throw(ArgumentError("empty input"))
	return svd_add(ρs, alg.trunc)[1]
end
function add(ρs::Vector{<:CanonicalMPO}, alg::DMRG1)
	isempty(ρs) && throw(ArgumentError("empty input"))
	return add!(svdguess_add(ρs, alg.D), ρs, alg)
end
add(ψA::CanonicalMPS, ψB::CanonicalMPS, alg::SVDCompression=DefaultMultAlg) = add([ψA, ψB], alg)
add(ρA::CanonicalMPO, ρB::CanonicalMPO, alg::SVDCompression=DefaultMultAlg) = add([ρA, ρB], alg)
add(ψA::CanonicalMPS, ψB::CanonicalMPS, alg::DMRG1) = add([ψA, ψB], alg)
add(ρA::CanonicalMPO, ρB::CanonicalMPO, alg::DMRG1) = add([ρA, ρB], alg)

# ---------- two-site (DMRG2) sweeps and interface ----------

# the two-site target of the sum: the accumulated overlap blocks of all chains
function _reduce_two_site(m::AddCache, s::Integer)
	acc = missing
	for c in m.hstorage
		t = _reduce_two_site(c, s)
		acc = acc isa Missing ? t : acc + t
	end
	acc isa Missing && throw(ArgumentError("empty AddCache"))
	return acc
end

function leftsweep!(m::AddCache, alg::DMRG2)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		t2 = _reduce_two_site(m, s)
		kvals[s] = norm(t2)
		(alg.verbosity > 2) && _logupdate(stdout, "l2r", s, kvals[s])
		_als2_update!(m.bra, s, t2, alg; move_right=true)
		for c in m.hstorage
			updateleft!(c, s)
		end
	end
	kvals[L] = kvals[L-1]
	return kvals
end

function rightsweep!(m::AddCache, alg::DMRG2)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		t2 = _reduce_two_site(m, s - 1)
		kvals[k] = norm(t2)
		(alg.verbosity > 2) && _logupdate(stdout, "r2l", s - 1, kvals[k])
		k += 1
		_als2_update!(m.bra, s - 1, t2, alg; move_right=false)
		for c in m.hstorage
			updateright!(c, s)
		end
	end
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(m::AddCache, alg::DMRG2) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

# the KKT eigenvalue of the converged ALS direction: (linear form) / ⟨dir|dir⟩
_kkt_ratio(form::Number, dirnorm2::Number) = form / dirnorm2

function add!(out, chains, alg::DMRG2)
	cache = AddCache(out, chains)
	iterative_compute!(cache, alg)
	# Σ kets = β·bra at convergence: β = Σ_n ⟨bra|ket_n⟩ / ⟨bra|bra⟩. Unlike the other
	# DMRG2 drivers, `add` restores the physical scale of the (cancellation-prone) sum
	# from the problem's KKT eigenvalue; `setscaling!` first, `lmul!` second (the
	# folding `lmul!` must not be overwritten by a later `setscaling!`)
	setscaling!(out, scaling(chains[1]))
	form = sum(_dot(out, c) for c in chains)
	lmul!(_kkt_ratio(form, _dot(out, out)), out)
	return out
end

function add(ψs::Vector{<:CanonicalMPS}, alg::DMRG2)
	isempty(ψs) && throw(ArgumentError("empty input"))
	return add!(svdguess_add(ψs, _guess_bond(alg.trunc)), ψs, alg)
end
function add(ρs::Vector{<:CanonicalMPO}, alg::DMRG2)
	isempty(ρs) && throw(ArgumentError("empty input"))
	return add!(svdguess_add(ρs, _guess_bond(alg.trunc)), ρs, alg)
end
add(ψA::CanonicalMPS, ψB::CanonicalMPS, alg::DMRG2) = add([ψA, ψB], alg)
add(ρA::CanonicalMPO, ρB::CanonicalMPO, alg::DMRG2) = add([ρA, ρB], alg)
