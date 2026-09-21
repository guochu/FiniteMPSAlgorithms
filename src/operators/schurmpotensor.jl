# sparse MPO site tensors: an MPOTensor stored as a matrix of small operators.
#
# AbstractSparseMPOTensor defines the shared interface; two storages are provided:
#   * SparseMPOTensor: a general matrix of local operators (no block structure)
#   * SchurMPOTensor: the Jordan/Schur upper-triangular form, stored as four blocks
#     A/B/C/D with implicit identity corners, mirroring MPSKit's JordanMPOTensor.

abstract type AbstractSparseMPOTensor end

# ---------- shared interface ----------

phydim(m::AbstractSparseMPOTensor) = m.d
ophydim(m::AbstractSparseMPOTensor) = m.d
iphydim(m::AbstractSparseMPOTensor) = m.d

Base.lastindex(m::AbstractSparseMPOTensor) = prod(size(m))
Base.lastindex(m::AbstractSparseMPOTensor, i::Int) = size(m, i)

"""
	_rawelement(W, i, j)

The raw (unexpanded) element stored at logical block position `(i, j)`: a local
operator matrix, a scalar (denoting a proportional-to-identity block), or `nothing`
for a structurally absent slot. Concrete sparse tensors implement this.
"""
_rawelement(::AbstractSparseMPOTensor, ::Int, ::Int) = nothing

_expand_element(x::Number, d::Int) =
	iszero(x) ? zeros(typeof(x), d, d) : x * isometry(typeof(x), d)
_expand_element(x::AbstractMatrix, ::Int) = x

# dense d×d block access: scalars expand to zeros / scalar·isometry, absent slots to zeros
function Base.getindex(m::AbstractSparseMPOTensor, i::Int, j::Int)
	x = _rawelement(m, i, j)
	x === nothing && return zeros(scalartype(m), phydim(m), phydim(m))
	return _expand_element(x, phydim(m))
end

Base.keys(x::AbstractSparseMPOTensor) =
	Iterators.filter(a -> contains(x, a[1], a[2]),
		Iterators.product(1:size(x, 1), 1:size(x, 2)))
opkeys(x::AbstractSparseMPOTensor) =
	Iterators.filter(a -> !isscal(x, a[1], a[2]), keys(x))
scalkeys(x::AbstractSparseMPOTensor) =
	Iterators.filter(a -> isscal(x, a[1], a[2]), keys(x))

Base.contains(x::AbstractSparseMPOTensor, i::Int, j::Int) =
	((xe = _rawelement(x, i, j); xe !== nothing && xe != zero(scalartype(x))))
isscal(x::AbstractSparseMPOTensor, i::Int, j::Int) =
	((xe = _rawelement(x, i, j); xe isa Number && xe != zero(scalartype(x))))

"""
	isid(x::AbstractMatrix)

Check whether the given local operator is proportional to the identity;
returns `(is_identity, proportionality_scalar)`.
"""
isid(x::Number; kwargs...) = true, x
function isid(x::AbstractMatrix; kwargs...)
	id = isometry(scalartype(x), size(x, 1))
	return _is_prop_util(x, id; kwargs...)
end
function _is_prop_util(x, a; atol::Real=1.0e-14)
	scal = dot(a, x) / dot(a, a)
	diff = x - scal * a
	scal = (scal ≈ 0.0) ? 0.0 : scal
	return norm(diff) < atol, scal
end

function compute_scalartype(data::AbstractMatrix)
	T = Float64
	for sj in data
		if isa(sj, Number) || isa(sj, AbstractMatrix)
			T = promote_type(T, scalartype(sj))
		else
			throw(ArgumentError("elements must be scalar or Matrix"))
		end
	end
	return T
end

# validate a matrix of local operators, compress proportional-to-identity blocks to
# scalars and return the homogeneous element matrix plus the physical dimension
# the same row should have the same left space; the same column the same right space
function compute_mpotensor_data(::Type{M}, ::Type{T}, data::AbstractMatrix) where {M<:AbstractMatrix, T<:Number}
	@assert !isempty(data)
	m, n = size(data)
	new_data = Array{Union{M, T}, 2}(undef, m, n)

	d::Union{Nothing, Int} = nothing
	for i in 1:m, j in 1:n
		sj = data[i, j]
		if isa(sj, AbstractMatrix)
			(size(sj, 1) == size(sj, 2)) || throw(DimensionMismatch("physical space mismatch"))
			s_p = size(sj, 1)
			if isnothing(d)
				d = s_p
			else
				(d == s_p) || throw(DimensionMismatch("physical space mismatch"))
			end
		end
	end
	isnothing(d) && throw(ArgumentError("pspace is missing"))
	for i in 1:m, j in 1:n
		sj = data[i, j]
		if isa(sj, AbstractMatrix)
			_is_id, scal = isid(sj)
			_is_id && (sj = scal)
		end
		if isa(sj, AbstractMatrix)
			sj = convert(M, sj)
		else
			isa(sj, Number) || throw(ArgumentError("elt should either be a tensor or a scalar"))
			sj = convert(T, sj)
		end
		new_data[i, j] = sj
	end
	return new_data, d
end

# convert a setindex! input into a storable element: scalar -> T, d×d matrix -> M
function _as_element(::Type{M}, ::Type{T}, v, d::Int) where {M<:AbstractMatrix, T<:Number}
	if v isa Number
		return convert(T, v)
	elseif v isa AbstractMatrix
		(size(v, 1) == size(v, 2) == d) ||
			throw(DimensionMismatch("input matrix size mismatch with phydim"))
		return convert(M, v)
	end
	return throw(ArgumentError("input should be scalar or Matrix type"))
end

_lmul_elements!(f::Number, A::AbstractArray) = (for I in eachindex(A)
	A[I] = f * A[I]
end; A)

# ---------- SparseMPOTensor: general matrix form ----------

"""
	SparseMPOTensor{M<:AbstractMatrix, T<:Number} <: AbstractSparseMPOTensor

A general sparse MPOTensor: a (non-triangular) matrix whose entries are either local
`d×d` operators or scalars (scalars denote proportional-to-identity blocks).
"""
struct SparseMPOTensor{M<:AbstractMatrix, T<:Number} <: AbstractSparseMPOTensor
	Os::Array{Union{M, T}, 2}
	d::Int
end

SparseMPOTensor{M, T}(data::AbstractMatrix) where {M<:AbstractMatrix, T<:Number} =
	SparseMPOTensor{M, T}(compute_mpotensor_data(M, T, data)...)
function SparseMPOTensor(data::AbstractMatrix)
	T = compute_scalartype(data)
	return SparseMPOTensor{Matrix{T}, T}(data)
end
function SparseMPOTensor(data::AbstractMatrix{Union{M, T}}) where {M<:AbstractMatrix, T<:Number}
	# a homogeneous Matrix{M} resolves the diagonal with T = Union{}; recover the
	# scalar type from the matrix element type
	T2 = T === Union{} ? scalartype(M) : T
	return SparseMPOTensor{M, T2}(data)
end
# direct construction from a homogeneous element matrix uses the default struct
# constructor, e.g. SparseMPOTensor{M,T}(Os, d)

Base.size(m::SparseMPOTensor) = size(m.Os)
Base.size(m::SparseMPOTensor, i::Int) = size(m.Os, i)
_rawelement(m::SparseMPOTensor, i::Int, j::Int) = m.Os[i, j]

Base.copy(x::SparseMPOTensor) = SparseMPOTensor(copy(x.Os), phydim(x))
scalartype(::Type{SparseMPOTensor{M, T}}) where {M, T} = T
function Base.complex(x::SparseMPOTensor{M, T}) where {M, T}
	TC = complex(T)
	return SparseMPOTensor{Matrix{TC}, TC}(complex.(x.Os), phydim(x))
end

function Base.setindex!(m::SparseMPOTensor{M, T}, v, i::Int, j::Int) where {M, T}
	m.Os[i, j] = _as_element(M, T, v, phydim(m))
	return m
end

# ---------- SchurMPOTensor: Jordan upper-triangular block form ----------
#
# logical block matrix (m rows × n columns):
#
#   [ 1  C  D
#     0  A  B
#     0  0  1 ]
#
# A is the interior×interior block, B the interior→closing column, C the vacuum→interior
# row and D the vacuum→closing corner; the two identity corners are implicit and are
# not stored. Edge tensors (first site 1×n keeps the vacuum row; last site n×1 keeps
# the closing column) arise from the same structure with empty interior blocks.

"""
	SchurMPOTensor{M<:AbstractMatrix, T<:Number} <: AbstractSparseMPOTensor

Upper triangular Jordan/Schur block form of an MPO site tensor, stored as the four
blocks `A` (interior), `B` (interior→closing), `C` (vacuum→interior) and `D`
(vacuum→closing), with implicit identity corners — MPSKit `JordanMPOTensor` style.
"""
struct SchurMPOTensor{M<:AbstractMatrix, T<:Number} <: AbstractSparseMPOTensor
	V::NTuple{2, Int}
	A::Array{Union{M, T}, 2}
	B::Array{Union{M, T}, 1}
	C::Array{Union{M, T}, 1}
	D::Base.RefValue{Union{M, T}}
	d::Int
end

# interior block counts belonging to a logical shape
_interior_counts(V::NTuple{2, Int}) = (V[1] == 1 ? 0 : V[1] - 2, V[2] == 1 ? 0 : V[2] - 2)

# full data constructor (validated, MPSKit-style)
function SchurMPOTensor{M, T}(d::Int, V::NTuple{2, Int},
		A::AbstractMatrix, B::AbstractVector, C::AbstractVector, D) where {M<:AbstractMatrix, T<:Number}
	nr, nc = _interior_counts(V)
	size(A) == (nr, nc) ||
		throw(DimensionMismatch("A block has size $(size(A)), expected ($nr, $nc)"))
	length(B) == nr ||
		throw(DimensionMismatch("B block has length $(length(B)), expected $nr"))
	length(C) == nc ||
		throw(DimensionMismatch("C block has length $(length(C)), expected $nc"))
	De = D isa Number ? convert(T, D) : convert(M, D)
	# build via the default field-order constructor (this outer method is d-first)
	return SchurMPOTensor{M, T}(V, A, B, C, Base.RefValue{Union{M, T}}(De), d)
end

# zero-initialized tensor: every slot absent (scalar zero)
function SchurMPOTensor{M, T}(d::Int, V::NTuple{2, Int}) where {M<:AbstractMatrix, T<:Number}
	nr, nc = _interior_counts(V)
	z = zero(T)
	A = fill!(Array{Union{M, T}, 2}(undef, nr, nc), z)
	B = fill!(Array{Union{M, T}, 1}(undef, nr), z)
	C = fill!(Array{Union{M, T}, 1}(undef, nc), z)
	return SchurMPOTensor{M, T}(d, V, A, B, C, z)
end

# construct from a logical block matrix: compress identities, verify the Jordan
# structure, then split into the four blocks
function SchurMPOTensor{M, T}(data::AbstractMatrix) where {M<:AbstractMatrix, T<:Number}
	Os, d = compute_mpotensor_data(M, T, data)
	return _schur_from_logical(Os, d)
end
"""
	SchurMPOTensor(data::AbstractMatrix)

Construct from a matrix whose entries are either `d×d` local operators or scalars.
Identity operators are compressed to their scalar; the two identity corners are
implicit and need not be supplied.
"""
SchurMPOTensor(data::AbstractMatrix) = SchurMPOTensor(compute_scalartype(data), data)
SchurMPOTensor(::Type{T}, data::AbstractMatrix) where {T<:Number} =
	SchurMPOTensor{Matrix{T}, T}(data)

function _schur_from_logical(Os::Array{Union{M, T}, 2}, d::Int) where {M<:AbstractMatrix, T<:Number}
	m, n = size(Os)
	nr, nc = _interior_counts((m, n))

	# implicit identity corners: if supplied, the entry must be absent or exactly identity
	_check_corner(x) = (x isa Number && (iszero(x) || isone(x))) ||
		throw(ArgumentError("identity corner of SchurMPOTensor cannot be set to $x"))
	n > 1 && _check_corner(Os[1, 1])
	m > 1 && _check_corner(Os[m, n])

	# vacuum column (col 1, unless it is the single closing column at the last site)
	# carries only the (1,1) identity
	if n > 1
		for i in 2:m
			Os[i, 1] == zero(T) ||
				throw(ArgumentError("SchurMPOTensor should be upper triangular"))
		end
	end
	# closing row (row m, unless it is the single vacuum row at the first site)
	# carries only the (m,n) identity
	if m > 1
		for j in 1:n-1
			Os[m, j] == zero(T) ||
				throw(ArgumentError("SchurMPOTensor should be upper triangular"))
		end
	end

	A = Array{Union{M, T}, 2}(undef, nr, nc)
	B = Array{Union{M, T}, 1}(undef, nr)
	C = Array{Union{M, T}, 1}(undef, nc)
	for a in 1:nr
		B[a] = Os[a+1, n]
		for b in 1:nc
			# interior block: entries below the A diagonal violate upper triangularity
			(a > b && Os[a+1, b+1] != zero(T)) &&
				throw(ArgumentError("SchurMPOTensor should be upper triangular"))
			A[a, b] = Os[a+1, b+1]
		end
	end
	for b in 1:nc
		C[b] = Os[1, b+1]
	end
	return SchurMPOTensor{M, T}(d, (m, n), A, B, C, Os[1, n])
end

Base.size(W::SchurMPOTensor) = W.V
Base.size(W::SchurMPOTensor, i::Int) = W.V[i]
scalartype(::Type{SchurMPOTensor{M, T}}) where {M, T} = T

# D is kept in a Ref (mutable slot without a mutable struct); present it as a plain field
function Base.getproperty(W::SchurMPOTensor, s::Symbol)
	s === :D && return getfield(W, :D)[]
	return getfield(W, s)
end
function Base.setproperty!(W::SchurMPOTensor, s::Symbol, v)
	if s === :D
		getfield(W, :D)[] = v isa Number ? convert(scalartype(W), v) : v
		return v
	end
	return setfield!(W, s, v)
end

function _rawelement(W::SchurMPOTensor{M, T}, i::Int, j::Int) where {M, T}
	m, n = W.V
	if i == 1
		j == n && return getfield(W, :D)[]
		(1 < j < n) && return W.C[j-1]
		(n > 1 && j == 1) && return one(T)
	elseif 1 < i < m
		j == n && return W.B[i-1]
		(1 < j < n) && return W.A[i-1, j-1]
	elseif i == m
		(m > 1 && j == n) && return one(T)
	end
	return nothing
end

function Base.setindex!(W::SchurMPOTensor{M, T}, v, i::Int, j::Int) where {M, T}
	m, n = W.V
	(1 <= i <= m && 1 <= j <= n) || throw(BoundsError(W, (i, j)))
	x = _as_element(M, T, v, phydim(W))
	if i == 1 && j == n
		getfield(W, :D)[] = x
	elseif i == 1 && 1 < j < n
		W.C[j-1] = x
	elseif 1 < i < m && j == n
		W.B[i-1] = x
	elseif 1 < i < m && 1 < j < n
		(i > j) && throw(ArgumentError("not allowed to set the lower triangular portion"))
		W.A[i-1, j-1] = x
	else
		throw(ArgumentError("cannot set index ($i, $j): implicit identity corner or structural zero"))
	end
	return W
end

_copy_orelse(x::Number) = x
_copy_orelse(x::AbstractMatrix) = copy(x)
Base.copy(W::SchurMPOTensor{M, T}) where {M, T} =
	SchurMPOTensor{M, T}(phydim(W), W.V, copy(W.A), copy(W.B), copy(W.C),
		_copy_orelse(getfield(W, :D)[]))

function Base.complex(W::SchurMPOTensor{M, T}) where {M, T}
	TC = complex(T)
	return SchurMPOTensor{Matrix{TC}, TC}(phydim(W), W.V,
		complex.(W.A), complex.(W.B), complex.(W.C), complex(getfield(W, :D)[]))
end

# expand the logical Jordan form into a general SparseMPOTensor (identity corners explicit)
function Base.convert(::Type{SparseMPOTensor}, W::SchurMPOTensor{M, T}) where {M, T}
	m, n = W.V
	data = fill!(Array{Union{M, T}, 2}(undef, m, n), zero(T))
	(n > 1) && (data[1, 1] = one(T))
	(m > 1) && (data[m, n] = one(T))
	data[1, n] = getfield(W, :D)[]
	for b in eachindex(W.C)
		data[1, b+1] = W.C[b]
	end
	for a in eachindex(W.B)
		data[a+1, n] = W.B[a]
	end
	for a in axes(W.A, 1), b in axes(W.A, 2)
		data[a+1, b+1] = W.A[a, b]
	end
	return SparseMPOTensor{M, T}(data, phydim(W))
end

function LinearAlgebra.lmul!(f::Number, W::SchurMPOTensor)
	_lmul_elements!(f, W.A)
	_lmul_elements!(f, W.B)
	_lmul_elements!(f, W.C)
	D = getfield(W, :D)
	D[] = f * D[]
	return W
end
