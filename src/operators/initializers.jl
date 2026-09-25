# initializers for operator chains

"""
	identitympo(::Type{T}, ds) -> MPO

The identity operator chain with bond dimension 1 (equivalent to `MPO(T, ds)`).
"""
identitympo(::Type{T}, ds::AbstractVector{Int}) where {T<:Number} = MPO(T, ds)

"""
	prodmpo(::Type{T}, ds, positions, ops) -> MPO
	prodmpo(::Type{T}, ds, position, op) -> MPO

A bond-dimension-1 product chain for one operator term: `ops[i]` acts on `positions[i]`
(ascending, possibly with gaps — gap sites carry the identity), all other sites carry the
identity. Use the exact `+`/`*` on `MPO` to assemble general Hamiltonians.
"""
function prodmpo(::Type{T}, ds::AbstractVector{Int}, positions::AbstractVector{<:Integer},
				 ops::AbstractVector{<:AbstractMatrix}) where {T<:Number}
	(length(positions) == length(ops)) || throw(DimensionMismatch())
	issorted(positions) || throw(ArgumentError("positions must be ascending"))
	data = Vector{Array{T,4}}(undef, length(ds))
	k = 1
	for (i, d) in enumerate(ds)
		if k <= length(positions) && positions[k] == i
			op = ops[k]
			(size(op) == (d, d)) || throw(DimensionMismatch("operator on site $i must be $d×$d"))
			data[i] = reshape(convert(Matrix{T}, op), 1, d, 1, d)
			k += 1
		else
			data[i] = reshape(isometry(T, d), 1, d, 1, d)
		end
	end
	return MPO(data)
end
prodmpo(::Type{T}, ds::AbstractVector{Int}, position::Integer, op::AbstractMatrix) where {T<:Number} =
	prodmpo(T, ds, [position], [op])

"""
	changebond!(h::CanonicalMPO; D=Defaults.D, noise=1e-10) -> h

Bring the bond profile of `h` to `min(D, feasible)`: bonds larger than their target are
shrunk by slicing the leading bond indices (the first rows/columns of the bond space;
singular values are ignored), smaller bonds are grown with random entries of magnitude
`noise` (a small perturbation of the represented operator). The Schmidt values do not
survive the resize: they are unset, and the gauge is left as produced by the resize —
if a specific canonical form is needed, call [`canonicalize!`](@ref) afterwards. Used
to prepare initial guesses of the ALS operator products.
"""
function changebond!(h::CanonicalMPO; D::Int=Defaults.D, noise::Real=1e-10)
	_changebond!(h; D, noise)
	unset_svectors!(h)
	return h
end

# raw bond-profile refit of the chain data (shared with the ALS guess preparation, e.g.
# `seq2seq!`): slicing/noise padding never re-gauges, plain MPO data stays untouched
function _changebond!(h::AbstractMPO; D::Int=Defaults.D, noise::Real=1e-10)
	isempty(h.data) && return h
	T = scalartype(h)
	L = length(h)
	ds_out = ophydims(h)
	ds_in = iphydims(h)
	Dl = ones(Int, L + 1)
	for i in 1:L
		Dl[i+1] = min(D, Dl[i] * ds_out[i] * ds_in[i])
	end
	Dr = ones(Int, L + 1)
	for i in L:-1:1
		Dr[i] = min(D, Dr[i+1] * ds_out[i] * ds_in[i])
	end
	b = min.(Dl, Dr)
	newdata = Vector{Array{T,4}}(undef, L)
	for i in 1:L
		dl = i == 1 ? 1 : b[i]
		dr = i == L ? 1 : b[i+1]
		t = noise * randn(T, dl, ds_out[i], dr, ds_in[i])
		vr = 1:min(dl, size(h[i], 1))
		vc = 1:min(dr, size(h[i], 3))
		t[vr, :, vc, :] = h[i][vr, :, vc, :]
		newdata[i] = t
	end
	copy!(h.data, newdata)
	return h
end
