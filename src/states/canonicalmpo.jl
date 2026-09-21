# CanonicalMPO: canonical-form 4-index chain with Schmidt values, layout consistent with TEMPO
# ProcessTensor. Semantics: a density matrix (mixed state) / process tensor.
# site tensor: W[aL, p_out, aR, p_in] :: Array{T,4}, vacuum boundaries (dimension 1)

"""
    CanonicalMPO{T<:Number, R<:Real} <: AbstractMPO{T}

One-dimensional canonical tensor network representing a density matrix (process tensor).
Site tensor conventions:

- Dimension 1: left auxiliary (bond) index, dimension 1 (vacuum) at the leftmost site
- Dimension 2: output physical index
- Dimension 3: right auxiliary (bond) index, dimension 1 (vacuum) at the rightmost site
- Dimension 4: input physical index

Fields `s` and `scaling` follow the same conventions as [`CanonicalMPS`](@ref).
"""
struct CanonicalMPO{T<:Number, R<:Real} <: AbstractMPO{T}
	data::Vector{Array{T, 4}}
	s::Vector{Union{Missing, Vector{R}}}
	scaling::Ref{Float64}

	function CanonicalMPO{T, R}(data::AbstractVector, s::AbstractVector, scaling::Ref{<:Real}) where {T<:Number, R<:Real}
		_check_mpo_space(data)
		(length(data) + 1 == length(s)) ||
			throw(DimensionMismatch("length of Schmidt values must be length of site tensors + 1"))
		s2 = [ismissing(sv) ? missing : convert(Vector{R}, collect(sv)) for sv in s]
		new{T, R}(convert(Vector{Array{T, 4}}, data), s2, scaling)
	end
end

CanonicalMPO{T, R}(data::AbstractVector, s::AbstractVector, scaling::Real=1.0) where {T<:Number, R<:Real} =
	CanonicalMPO{T, R}(data, s, Ref(float(scaling)))
CanonicalMPO{T, R}(data::AbstractVector, scaling::Real=1.0) where {T<:Number, R<:Real} =
	CanonicalMPO{T, R}(data, _default_mpo_s(data, R), scaling)

CanonicalMPO(data::AbstractVector{<:MPOTensor{T}}, s::AbstractVector; scaling::Real=1.0) where {T<:Number} =
	CanonicalMPO{T, real(T)}(data, s, scaling)
CanonicalMPO(data::AbstractVector{<:MPOTensor{T}}, s::AbstractVector, scaling::Ref{<:Real}) where {T<:Number} =
	CanonicalMPO{T, real(T)}(data, s, scaling)
CanonicalMPO(data::AbstractVector{<:MPOTensor{T}}; scaling::Real=1.0) where {T<:Number} =
	CanonicalMPO{T, real(T)}(data, _default_mpo_s(data, real(T)), scaling)

"""
    CanonicalMPO(::Type{T}, ds) -> CanonicalMPO
    CanonicalMPO(::Type{T}, L; d=2)

The identity `CanonicalMPO` with vacuum boundaries: site tensors `reshape(isometry(T, d), 1, d, 1, d)`.
"""
CanonicalMPO(::Type{T}, ds::AbstractVector{Int}) where {T<:Number} =
	CanonicalMPO([reshape(isometry(T, d), 1, d, 1, d) for d in ds])
CanonicalMPO(::Type{T}, L::Int; d::Int=2) where {T<:Number} = CanonicalMPO(T, fill(d, L))

Base.copy(ρ::CanonicalMPO) = CanonicalMPO(copy(ρ.data), _copy_s(ρ), Ref(scaling(ρ)))
Base.copy!(dst::CanonicalMPO, src::CanonicalMPO) = (copy!(dst.data, src.data); copy!(dst.s, src.s); setscaling!(dst, scaling(src)); dst)
function Base.complex(ρ::CanonicalMPO{T, R}) where {T, R}
	T <: Complex && return copy(ρ)
	return CanonicalMPO(complex.(ρ.data), [ismissing(s) ? s : complex.(s) for s in ρ.s], Ref(scaling(ρ)))
end

function Base.show(io::IO, ρ::CanonicalMPO)
	print(io, "CanonicalMPO{", scalartype(ρ), "} with ", length(ρ), " sites, maxbond = ", bonddim(ρ),
		", scaling = ", scaling(ρ))
end

svectors_uninitialized(ρ::CanonicalMPO) = any(ismissing, ρ.s[2:end-1])
function unset_svectors!(ρ::CanonicalMPO)
	for i in 2:length(ρ.s)-1
		ρ.s[i] = missing
	end
	return ρ
end

function _default_mpo_s(data::AbstractVector{<:MPOTensor}, R::Type)
	s = Vector{Union{Missing, Vector{R}}}(undef, length(data) + 1)
	fill!(s, missing)
	s[1] = ones(R, space_l(data[1]))
	s[end] = ones(R, space_r(data[end]))
	return s
end

function _check_mpo_space(data::AbstractVector)
	isempty(data) && error("no input tensors.")
	for i in 1:length(data)-1
		(size(data[i], 3) == size(data[i+1], 1)) ||
			throw(DimensionMismatch("bond dimension mismatch between site $i and site $(i+1)"))
	end
	(size(data[1], 1) == 1 && size(data[end], 3) == 1) ||
		throw(ArgumentError("boundary bond dimensions must be 1"))
	return true
end
