# CanonicalMPS: canonical-form MPS with Schmidt values, layout consistent with TEMPO ADT
# site tensor: A[aL, p, aR] :: Array{T,3}

"""
    CanonicalMPS{T<:Number, R<:Real} <: AbstractMPS{T}

One-dimensional canonical tensor network representing a finite quantum state.
Site tensor conventions:

- Dimension 1: left auxiliary (bond) index, dimension 1 at the leftmost site
- Dimension 2: physical index
- Dimension 3: right auxiliary (bond) index, dimension 1 at the rightmost site

# Fields
- `data::Vector{Array{T,3}}`: list of site tensors
- `s::Vector{Union{Missing, Vector{R}}}`: Schmidt values; `s[b+1]` stores the spectrum at bond `b`
  (bond `b` lies between site `b` and `b+1`); `missing` when uninitialized
- `scaling::Ref{Float64}`: per-site overall scaling factor (total scaling = `scaling^L`)
"""
struct CanonicalMPS{T<:Number, R<:Real} <: AbstractMPS{T}
	data::Vector{Array{T, 3}}
	s::Vector{Union{Missing, Vector{R}}}
	scaling::Ref{Float64}

	function CanonicalMPS{T, R}(data::AbstractVector, s::AbstractVector, scaling::Ref{<:Real}) where {T<:Number, R<:Real}
		_check_mps_space(data)
		(length(data) + 1 == length(s)) ||
			throw(DimensionMismatch("length of Schmidt values must be length of site tensors + 1"))
		s2 = [ismissing(sv) ? missing : convert(Vector{R}, collect(sv)) for sv in s]
		new{T, R}(convert(Vector{Array{T, 3}}, data), s2, scaling)
	end
end

CanonicalMPS{T, R}(data::AbstractVector, s::AbstractVector, scaling::Real=1.0) where {T<:Number, R<:Real} =
	CanonicalMPS{T, R}(data, s, Ref(float(scaling)))
CanonicalMPS{T, R}(data::AbstractVector, scaling::Real=1.0) where {T<:Number, R<:Real} =
	CanonicalMPS{T, R}(data, _default_s(data, R), scaling)

CanonicalMPS(data::AbstractVector{<:MPSTensor{T}}, s::AbstractVector; scaling::Real=1.0) where {T<:Number} =
	CanonicalMPS{T, real(T)}(data, s, scaling)
CanonicalMPS(data::AbstractVector{<:MPSTensor{T}}, s::AbstractVector, scaling::Ref{<:Real}) where {T<:Number} =
	CanonicalMPS{T, real(T)}(data, s, scaling)
CanonicalMPS(data::AbstractVector{<:MPSTensor{T}}; scaling::Real=1.0) where {T<:Number} =
	CanonicalMPS{T, real(T)}(data, _default_s(data, real(T)), scaling)

"""
    CanonicalMPS(::Type{T}, ds) -> CanonicalMPS
    CanonicalMPS(::Type{T}, L; d=2)

A bond-dimension-1 `CanonicalMPS` of all-ones site tensors with physical dimensions `ds`.
"""
CanonicalMPS(::Type{T}, ds::AbstractVector{Int}) where {T<:Number} =
	CanonicalMPS([ones(T, 1, d, 1) for d in ds])
CanonicalMPS(::Type{T}, L::Int; d::Int=2) where {T<:Number} = CanonicalMPS(T, fill(d, L))

Base.copy(ψ::CanonicalMPS) = CanonicalMPS(copy(ψ.data), _copy_s(ψ), Ref(scaling(ψ)))
Base.copy!(dst::CanonicalMPS, src::CanonicalMPS) = (copy!(dst.data, src.data); copy!(dst.s, src.s); setscaling!(dst, scaling(src)); dst)
function Base.complex(ψ::CanonicalMPS{T, R}) where {T, R}
	T <: Complex && return copy(ψ)
	return CanonicalMPS(complex.(ψ.data), [ismissing(s) ? s : complex.(s) for s in ψ.s], Ref(scaling(ψ)))
end

function Base.show(io::IO, ψ::CanonicalMPS)
	print(io, "CanonicalMPS{", scalartype(ψ), "} with ", length(ψ), " sites, maxbond = ", bonddim(ψ),
		", scaling = ", scaling(ψ))
end

# Schmidt values: `ψ.s[i]` is the (missing or vector) spectrum at bond i-1

"""
    svectors_uninitialized(ψ) -> Bool

Whether any internal bond of `ψ` has uninitialized (`missing`) Schmidt values.
"""
svectors_uninitialized(ψ::CanonicalMPS) = any(ismissing, ψ.s[2:end-1])

"""
    unset_svectors!(ψ)

Reset all internal Schmidt values to `missing` (boundary bonds keep their trivial `ones(1)`).
"""
function unset_svectors!(ψ::CanonicalMPS)
	for i in 2:length(ψ.s)-1
		ψ.s[i] = missing
	end
	return ψ
end

function _default_s(data::AbstractVector{<:MPSTensor}, R::Type)
	s = Vector{Union{Missing, Vector{R}}}(undef, length(data) + 1)
	fill!(s, missing)
	s[1] = ones(R, space_l(data[1]))
	s[end] = ones(R, space_r(data[end]))
	return s
end

function _check_mps_space(data::AbstractVector)
	isempty(data) && error("no input tensors.")
	for i in 1:length(data)-1
		(size(data[i], 3) == size(data[i+1], 1)) ||
			throw(DimensionMismatch("bond dimension mismatch between site $i and site $(i+1)"))
	end
	(size(data[1], 1) == 1 && size(data[end], 3) == 1) ||
		throw(ArgumentError("boundary bond dimensions must be 1"))
	return true
end
