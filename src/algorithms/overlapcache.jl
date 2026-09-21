# OverlapCache: the generic "problem" carrier of all iterative (ALS) algorithms;
# bra is the side to be optimized, ket the fixed input side (brought to
# right-canonical form upon construction); cstorage[s] = ⟨bra[1:s-1]|ket[1:s-1]⟩ :: Matrix

struct OverlapCache{A, B, T}
	bra::A
	ket::B
	cstorage::Vector{Matrix{T}}
end

function OverlapCache(ψA::CanonicalMPS, ψB::CanonicalMPS)
	(length(ψA) == length(ψB)) || throw(ArgumentError("dimension mismatch"))
	B = copy(ψB)
	rightorth!(B)
	L = length(ψB)
	T = promote_type(scalartype(ψA), scalartype(ψB))
	cs = Vector{Matrix{T}}(undef, L + 1)
	cs[1] = ones(T, 1, 1)
	cs[L+1] = ones(T, 1, 1)
	for s in L:-1:2
		cs[s] = _updateright(cs[s+1], ψA[s], B[s])
	end
	return OverlapCache(ψA, B, cs)
end

function OverlapCache(hA::AbstractMPO, hB::AbstractMPO)
	(length(hA) == length(hB)) || throw(ArgumentError("dimension mismatch"))
	B = copy(hB)
	# right-gauge the working ket (plain MPO chains are gauged only internally)
	_rightorth!(B, SVD(), DefaultTruncation, false, 0)
	L = length(hB)
	T = promote_type(scalartype(hA), scalartype(hB))
	cs = Vector{Matrix{T}}(undef, L + 1)
	cs[1] = ones(T, 1, 1)
	cs[L+1] = ones(T, 1, 1)
	for s in L:-1:2
		cs[s] = _updateright(cs[s+1], hA[s], B[s])
	end
	return OverlapCache(hA, B, cs)
end

"""
	updateleft!(c::OverlapCache, site)

Update the left overlap environment at `site+1`; the bra tensor at `site` must be up to date.
"""
function updateleft!(c::OverlapCache, site::Integer)
	c.cstorage[site+1] = _updateleft(c.cstorage[site], c.bra[site], c.ket[site])
	return c
end

"""
	updateright!(c::OverlapCache, site)

Update the right overlap environment at `site-1`.
"""
function updateright!(c::OverlapCache, site::Integer)
	c.cstorage[site] = _updateright(c.cstorage[site+1], c.bra[site], c.ket[site])
	return c
end
