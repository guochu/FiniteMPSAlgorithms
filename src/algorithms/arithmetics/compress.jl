# compress: the unique exported interface for (truncating) compression of a single chain
#   compress(ψ::CanonicalMPS; alg) -> (CanonicalMPS, err)
#   compress(h::AbstractMPO; alg)  -> (AbstractMPO, err)
# dispatched on alg: SVDCompression (canonicalize with truncation) or DMRG1 (ALS maximizing
# the overlap between the compressed chain and the input chain)

# ALS compression cache: maximize |⟨out|inp⟩|; reuses OverlapCache
const CompressCache = OverlapCache

function leftsweep!(c::OverlapCache, alg::DMRG1)
	L = length(c.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		t = _reduce_compress_site(c, s)
		kvals[s] = norm(t)
		q, r = _gauge_left(t)
		c.bra[s] = q
		c.bra[s+1] = _contract_first(c.bra[s+1], r)
		updateleft!(c, s)
	end
	t = _reduce_compress_site(c, L)
	kvals[L] = norm(t)
	c.bra[L] = t
	return kvals
end

function rightsweep!(c::OverlapCache, alg::DMRG1)
	L = length(c.bra)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		t = _reduce_compress_site(c, s)
		kvals[k] = norm(t)
		k += 1
		l, q = _gauge_right(t)
		c.bra[s] = q
		c.bra[s-1] = _contract_last(c.bra[s-1], l)
		updateright!(c, s)
	end
	t = _reduce_compress_site(c, 1)
	kvals[L] = norm(t)
	c.bra[1] = t
	return kvals
end

sweep!(c::OverlapCache, alg::DMRG1) = vcat(leftsweep!(c, alg), rightsweep!(c, alg))

function _reduce_compress_site(c::OverlapCache, s::Integer)
	if c.ket[s] isa MPOTensor
		return @tensor tmp4[-1, -2, -3, -4] := c.cstorage[s][-1, 1] * c.ket[s][1, -2, 2, -4] * c.cstorage[s+1][-3, 2]
	end
	return @tensor tmp[-1, -2, -3] := c.cstorage[s][-1, 1] * c.ket[s][1, -2, 2] * c.cstorage[s+1][-3, 2]
end

# ---------- initial guess & cache construction ----------

"""
	svdguess_compress(x, D::Int)

A truncating SVD re-compression of `x` (bond cap `D`) as an initial guess for the
iterative `compress` (accurate but costly compared to zero padding).
"""
function svdguess_compress(x, D::Int)
	out, _ = _canonicalize!(copy(x); alg=Orthogonalize(SVD(), truncdim(D), false))
	return x isa CanonicalMPS ? out : CanonicalMPO(out.data)
end

"""
	compress!(out, x, alg::DMRG1) -> out

Single-site variational (ALS) compression of `x` refined in place on the initial guess
`out`. The sweeps determine direction and magnitude together at the data level; the
input's external scale is attached through the per-site `scaling` field of the output
(a scaling^L power is never materialized).
"""
function compress!(out, x, alg::DMRG1)
	bonddim(out) != alg.D && changebond!(out; D=alg.D)
	cache = OverlapCache(out, x)
	iterative_compute!(cache, alg)
	x isa Union{CanonicalMPS, CanonicalMPO} && setscaling!(out, scaling(x))
	return out
end

"""
	compress(ψ::CanonicalMPS, alg=SVDCompression(DefaultTruncation)) -> CanonicalMPS
	compress(h::AbstractMPO, alg=...) -> CanonicalMPO
	compress(x, alg::DMRG1) -> chain

Compress a single chain: the SVD route canonicalizes a copy of the input with a
truncating SVD sweep; the variational (ALS) route draws a `svdguess_compress(x, D)`
initial guess and refines it by sweeps.
"""
function compress(ψ::CanonicalMPS, alg::SVDCompression=DefaultMultAlg)
	return _canonicalize!(copy(ψ); alg=Orthogonalize(SVD(), alg.trunc, false))[1]
end
function compress(h::AbstractMPO, alg::SVDCompression=DefaultMultAlg)
	out, _ = _canonicalize!(copy(h); alg=Orthogonalize(SVD(), alg.trunc, false))
	return CanonicalMPO(out.data)
end
# block-sparse chains: operator gauging is written on 4-index site tensors
compress(h::MPOHamiltonian, alg::SVDCompression=DefaultMultAlg) =
	compress(MPO(tompotensors(h)), alg)

function compress(x, alg::DMRG1)
	out = svdguess_compress(x, alg.D)
	compress!(out, x, alg)
	return out
end
compress(h::MPOHamiltonian, alg::DMRG1) =
	compress(MPO(tompotensors(h)), alg)
