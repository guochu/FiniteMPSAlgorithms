# MPO: a plain chain of 4-index tensors representing a quantum operator
# site tensor: W[aL, p_out, aR, p_in] :: Array{T,4}, left boundary dimension 1

"""
    MPO{T<:Number} <: AbstractMPO{T}

A finite matrix product operator: a chain of rank-4 site tensors representing a quantum operator.
Site tensor conventions:

- Dimension 1: left auxiliary (bond) index, dimension 1 at the leftmost site
- Dimension 2: output physical index
- Dimension 3: right auxiliary (bond) index
- Dimension 4: input physical index
"""
struct MPO{T<:Number} <: AbstractMPO{T}
	data::Vector{Array{T, 4}}

	function MPO{T}(data::AbstractVector) where {T<:Number}
		_check_mpo_space(data)
		return new{T}(convert(Vector{Array{T, 4}}, data))
	end
end

MPO(data::AbstractVector{<:MPOTensor{T}}) where {T<:Number} = MPO{T}(data)

"""
    MPO(::Type{T}, ds) -> MPO

The identity `MPO`: site tensors `reshape(isometry(T, d), 1, d, 1, d)` with bond dimension 1.
"""
MPO(::Type{T}, ds::AbstractVector{Int}) where {T<:Number} =
	MPO([reshape(isometry(T, d), 1, d, 1, d) for d in ds])

MPO(h::CanonicalMPO) = MPO(h.data)

Base.copy(h::MPO) = MPO(copy(h.data))
Base.copy!(dst::MPO, src::MPO) = (copy!(dst.data, src.data); dst)
Base.complex(h::MPO{T}) where {T} = T <: Complex ? copy(h) : MPO(complex.(h.data))

function Base.show(io::IO, h::MPO)
	print(io, "MPO{", scalartype(h), "} with ", length(h), " sites, maxbond = ", bonddim(h))
end
