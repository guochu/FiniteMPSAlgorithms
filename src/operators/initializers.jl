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
	increase_bond!(h::AbstractMPO; D=Defaults.D) -> h

Grow the bond dimensions of `h` towards the profile `max_bonddims(ds, D)` by zero
padding (the operator itself is unchanged; a plain `MPO` is never re-gauged in place).
Used to seed the initial guess of the ALS operator products.
"""
function increase_bond!(h::AbstractMPO; D::Int=Defaults.D)
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
	for i in 1:L-1
		b[i+1] = max(b[i+1], bonddim(h, i))
	end
	newdata = Vector{Array{T,4}}(undef, L)
	for i in 1:L
		dl = i == 1 ? 1 : b[i]
		dr = i == L ? 1 : b[i+1]
		t = zeros(T, dl, ds_out[i], dr, ds_in[i])
		t[1:size(h[i], 1), :, 1:size(h[i], 3), :] = h[i]
		newdata[i] = t
	end
	copy!(h.data, newdata)
	return h
end
