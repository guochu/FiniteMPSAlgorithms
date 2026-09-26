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
function compute_mpotensor_data(::Type{T}, data::AbstractMatrix) where {T<:Number}
	@assert !isempty(data)
	m, n = size(data)
	new_data = Array{Union{Matrix{T}, T}, 2}(undef, m, n)

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
	isnothing(d) && throw(ArgumentError("cannot infer phydim: the logical matrix contains no operator " *
										"(all-scalar input; a site tensor needs at least one d×d block, " *
										"e.g. a site with no operator support has an all-scalar block matrix)"))
	for i in 1:m, j in 1:n
		sj = data[i, j]
		if isa(sj, AbstractMatrix)
			_is_id, scal = isid(sj)
			_is_id && (sj = scal)
		end
		if isa(sj, AbstractMatrix)
			sj = convert(Matrix{T}, sj)
		else
			isa(sj, Number) || throw(ArgumentError("elt should either be a tensor or a scalar"))
			sj = convert(T, sj)
		end
		new_data[i, j] = sj
	end
	return new_data, d
end

# convert a setindex! input into a storable element: scalar -> T, d×d matrix -> Matrix{T}
function _as_element(::Type{T}, v, d::Int) where {T<:Number}
	if v isa Number
		return convert(T, v)
	elseif v isa AbstractMatrix
		(size(v, 1) == size(v, 2) == d) ||
			throw(DimensionMismatch("input matrix size mismatch with phydim"))
		return convert(Matrix{T}, v)
	end
	return throw(ArgumentError("input should be scalar or Matrix type"))
end

_lmul_elements!(f::Number, A::AbstractArray) = (for I in eachindex(A)
	A[I] = f * A[I]
end; A)

# ---------- SparseMPOTensor: general matrix form ----------

"""
	SparseMPOTensor{T<:Number} <: AbstractSparseMPOTensor

A general sparse MPOTensor: a (non-triangular) matrix whose entries are either local
`d×d` operators (`Matrix{T}`) or scalars (scalars denote proportional-to-identity blocks).
"""
struct SparseMPOTensor{T<:Number} <: AbstractSparseMPOTensor
	Os::Array{Union{Matrix{T}, T}, 2}
	d::Int
end

SparseMPOTensor{T}(data::AbstractMatrix) where {T<:Number} =
	SparseMPOTensor{T}(compute_mpotensor_data(T, data)...)
function SparseMPOTensor(data::AbstractMatrix)
	T = compute_scalartype(data)
	return SparseMPOTensor{T}(data)
end
# direct construction from a homogeneous element matrix uses the default struct
# constructor, e.g. SparseMPOTensor{T}(Os, d)

Base.size(m::SparseMPOTensor) = size(m.Os)
Base.size(m::SparseMPOTensor, i::Int) = size(m.Os, i)
_rawelement(m::SparseMPOTensor, i::Int, j::Int) = m.Os[i, j]

Base.copy(x::SparseMPOTensor) = SparseMPOTensor(copy(x.Os), phydim(x))
scalartype(::Type{SparseMPOTensor{T}}) where {T} = T
function Base.complex(x::SparseMPOTensor{T}) where {T}
	TC = complex(T)
	return SparseMPOTensor{TC}(complex.(x.Os), phydim(x))
end

function Base.setindex!(m::SparseMPOTensor{T}, v, i::Int, j::Int) where {T}
	m.Os[i, j] = _as_element(T, v, phydim(m))
	return m
end

# ---------- SchurMPOTensor: Jordan upper-triangular block form ----------
#
# logical block matrix (m rows × n columns, always m, n >= 2):
#
#   [ 1  C  D
#     0  A  B
#     0  0  1 ]
#
# A is the interior×interior block, B the interior→closing column, C the vacuum→interior
# row and D the vacuum→closing corner; the two identity corners are implicit and are
# not stored. SchurMPOTensor does not handle chain boundaries — the logical shape is
# always size(W.A) .+ 2 (so space_l/space_r > 1); the vacuum row / closing column of
# the boundary sites are selected downstream by tompotensors.

"""
	SchurMPOTensor{T<:Number} <: AbstractSparseMPOTensor

Upper triangular Jordan/Schur block form of an MPO site tensor, stored as the four
blocks `A` (interior), `B` (interior→closing), `C` (vacuum→interior) and `D`
(vacuum→closing), with implicit identity corners — MPSKit `JordanMPOTensor` style.
The logical shape is not stored: `size`/`space_l`/`space_r` are inferred from the
size of `A`. The `D` corner lives in a `RefValue` (`_D`) to keep the struct
immutable while the corner stays mutable; `getproperty` dereferences it, so `W.D`
reads and writes the corner value directly. Chain boundaries are not handled —
`space_l`/`space_r > 1` always; use `tompotensors` to select the boundary
channels of a finite chain.
"""
struct SchurMPOTensor{T<:Number} <: AbstractSparseMPOTensor
	A::Array{Union{Matrix{T}, T}, 2}
	B::Array{Union{Matrix{T}, T}, 1}
	C::Array{Union{Matrix{T}, T}, 1}
	_D::Base.RefValue{Union{Matrix{T}, T}}
	d::Int
end

# the logical shape is always size(W.A) .+ 2: interior counts plus the vacuum
# row and the closing column
Base.size(W::SchurMPOTensor) = (size(W.A, 1) + 2, size(W.A, 2) + 2)
Base.size(W::SchurMPOTensor, i::Int) = size(W.A, i) + 2

# construct from a logical block matrix: compress identities, verify the Jordan
# structure, then split into the four blocks (the logical shape is taken from
# `size(data)` — the constructors never take it explicitly)
function SchurMPOTensor{T}(data::AbstractMatrix) where {T<:Number}
	Os, d = compute_mpotensor_data(T, data)
	return _schur_from_logical(Os, d)
end
"""
	SchurMPOTensor(data::AbstractMatrix)

Construct from a matrix whose entries are either `d×d` local operators or scalars.
Identity operators are compressed to their scalar; the two identity corners are
implicit and need not be supplied.
"""
SchurMPOTensor(data::AbstractMatrix) = SchurMPOTensor{compute_scalartype(data)}(data)

function _schur_from_logical(Os::Array{Union{Matrix{T}, T}, 2}, d::Int) where {T<:Number}
	m, n = size(Os)
	(m >= 2 && n >= 2) || throw(ArgumentError(
		"SchurMPOTensor requires the full logical shape (m, n >= 2), got ($m, $n) — chain boundaries are handled by tompotensors"))
	nr, nc = m - 2, n - 2

	# implicit identity corners: if supplied, the entry must be absent or exactly identity
	_check_corner(x) = (x isa Number && (iszero(x) || isone(x))) ||
		throw(ArgumentError("identity corner of SchurMPOTensor cannot be set to $x"))
	_check_corner(Os[1, 1])
	_check_corner(Os[m, n])

	# vacuum column (col 1) carries only the (1,1) identity
	for i in 2:m
		Os[i, 1] == zero(T) ||
			throw(ArgumentError("SchurMPOTensor should be upper triangular"))
	end
	# closing row (row m) carries only the (m,n) identity
	for j in 1:n-1
		Os[m, j] == zero(T) ||
			throw(ArgumentError("SchurMPOTensor should be upper triangular"))
	end

	A = Array{Union{Matrix{T}, T}, 2}(undef, nr, nc)
	B = Array{Union{Matrix{T}, T}, 1}(undef, nr)
	C = Array{Union{Matrix{T}, T}, 1}(undef, nc)
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
	return SchurMPOTensor{T}(A, B, C, Base.RefValue{Union{Matrix{T}, T}}(Os[1, n]), d)
end

scalartype(::Type{SchurMPOTensor{T}}) where {T} = T

# D is kept in a Ref under the field name `_D` (mutable slot without a mutable struct);
# it is presented as a plain property: W.D reads/writes the dereferenced corner value
function Base.getproperty(W::SchurMPOTensor, s::Symbol)
	s === :D && return getfield(W, :_D)[]
	return getfield(W, s)
end
function Base.setproperty!(W::SchurMPOTensor, s::Symbol, v)
	if s === :D
		getfield(W, :_D)[] = v isa Number ? convert(scalartype(W), v) : v
		return v
	end
	return setfield!(W, s, v)
end

function _rawelement(W::SchurMPOTensor{T}, i::Int, j::Int) where {T}
	m, n = size(W)
	if i == 1
		j == n && return getfield(W, :_D)[]
		(1 < j < n) && return W.C[j-1]
		j == 1 && return one(T)
	elseif 1 < i < m
		j == n && return W.B[i-1]
		(1 < j < n) && return W.A[i-1, j-1]
	elseif i == m
		j == n && return one(T)
	end
	return nothing
end

function Base.setindex!(W::SchurMPOTensor{T}, v, i::Int, j::Int) where {T}
	m, n = size(W)
	(1 <= i <= m && 1 <= j <= n) || throw(BoundsError(W, (i, j)))
	x = _as_element(T, v, phydim(W))
	if i == 1 && j == n
		getfield(W, :_D)[] = x
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
Base.copy(W::SchurMPOTensor{T}) where {T} =
	SchurMPOTensor{T}(copy(W.A), copy(W.B), copy(W.C),
		Base.RefValue{Union{Matrix{T}, T}}(_copy_orelse(getfield(W, :_D)[])), phydim(W))

function Base.complex(W::SchurMPOTensor{T}) where {T}
	TC = complex(T)
	return SchurMPOTensor{TC}(complex.(W.A), complex.(W.B), complex.(W.C),
		Base.RefValue{Union{Matrix{TC}, TC}}(complex(getfield(W, :_D)[])), phydim(W))
end

# expand the logical Jordan form into a general SparseMPOTensor (identity corners explicit)
function Base.convert(::Type{SparseMPOTensor}, W::SchurMPOTensor{T}) where {T}
	m, n = size(W)
	data = fill!(Array{Union{Matrix{T}, T}, 2}(undef, m, n), zero(T))
	data[1, 1] = one(T)
	data[m, n] = one(T)
	data[1, n] = getfield(W, :_D)[]
	for b in eachindex(W.C)
		data[1, b+1] = W.C[b]
	end
	for a in eachindex(W.B)
		data[a+1, n] = W.B[a]
	end
	for a in axes(W.A, 1), b in axes(W.A, 2)
		data[a+1, b+1] = W.A[a, b]
	end
	return SparseMPOTensor{T}(data, phydim(W))
end

function LinearAlgebra.lmul!(f::Number, W::SchurMPOTensor)
	_lmul_elements!(f, W.A)
	_lmul_elements!(f, W.B)
	_lmul_elements!(f, W.C)
	_D = getfield(W, :_D)
	_D[] = f * _D[]
	return W
end
