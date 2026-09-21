# vendored from TEMPO/src/tensorops/tensorfactorizations.jl
# note: TensorOperations does not provide `permute`/`scalar` (only `tensorscalar`/`tensorcopy`),
# so these implementations are kept.

scalar(x::AbstractArray{T}) where {T<:Number} = only(x)

"""
    permute(m::AbstractArray, perm)

Return a view of `m` with its dimensions permuted according to `perm` (a `PermutedDimsArray`), without copying data.
"""
permute(m::AbstractArray, perm) = PermutedDimsArray(m, perm)
"""
    permute(m::AbstractArray, left, right)

Group the dimensions into `left` and `right` and place them in that order, equivalent to `permute(m, (left..., right...))`.
"""
permute(m::AbstractArray, left, right) = permute(m, (left..., right...))

function random_hermitian(::Type{T}, n::Int) where {T <: Number}
	m = randn(T, n, n)
	return m + m'
end

random_unitary(::Type{T}, n::Int) where {T <: Number} = exp(im .* random_hermitian(T, n))

function isometry(::Type{T}, m::Int, n::Int) where {T<:Number}
	r = zeros(T, m, n)
	for i in 1:min(m, n)
		r[i, i] = 1
	end
	return r
end
"""
    isometry(T, m, n)
    isometry(T, d)
    isometry(m, n)
    isometry(d)

Return the rectangular identity (isometry) matrix of size `m × n` (`d × d`) with element type `T` (default `Float64`).
"""
isometry(::Type{T}, d::Int) where {T<:Number} = isometry(T, d, d)
isometry(m::Int, n::Int) = isometry(Float64, m, n)
isometry(d::Int) = isometry(Float64, d)

function _group_extent(extent::NTuple{N,Int}, idx::NTuple{N1,Int}) where {N,N1}
	return ntuple(Val(N1)) do i
		l = sum(idx[j] for j in 1:(i - 1); init = 0)
		prod(ntuple(j -> extent[l + j], idx[i]))
	end
end


function tie(a::AbstractArray{T, N}, axs::NTuple{N1, Int}) where {T, N, N1}
	(sum(axs) != N) && error("total number of axes should equal to tensor rank.")
	return reshape(a, _group_extent(size(a), axs))
end

function Base.kron(a::AbstractArray{Ta, N}, b::AbstractArray{Tb, N}) where {Ta<:Number, Tb<:Number, N}
	N == 0 && error("empty tensors.")
	sa = size(a)
	sb = size(b)
	sc = ntuple(i->sa[i]*sb[i], Val(N))
	c = Array{promote_type(Ta, Tb), N}(undef, sc)
	for index in CartesianIndices(a)
		ranges = ntuple(j->((index[j]-1)*sb[j]+1):(index[j]*sb[j]), Val(N))
		c[ranges...] = a[index]*b
	end
	return c
end

"""
    tsvd!(a; trunc=NoTruncation(), alg=SDD())

Truncated singular value decomposition of the matrix `a`; returns `(u, s, v, err)`.

**Reconstruction convention (plain):** `a = u * Diagonal(s) * v` — the factors multiply
*without any further adjoint*: `v` is the adjoint `V†` of the right singular vectors,
already conjugated and stored as an ordinary array. With `trunc = NoTruncation()` the
decomposition is exact up to round-off; with truncation only the leading `r` singular
values are kept and `err = sqrt(Σ_{k>r} σk²)` is the discarded weight. The factors are

* `u` (`m × r`): orthonormal columns, `u' * u = I`;
* `v` (`r × n`): orthonormal rows, `v * v' = I` (the isometry `V†`);
* `s`: the kept singular values in descending order.

`alg` selects the SVD driver: `SDD()` (divide-and-conquer with safe fallback) or
`SVD()` (QR iteration, LAPACK `gesvd`).

Note: the input matrix `a` is used as workspace and may be **destroyed** in place;
pass a copy if the input must be preserved.
"""
function tsvd!(a::StridedMatrix; trunc::TruncationScheme=NoTruncation(), alg::Union{SVD,SDD}=SDD())
	u, S, v = svd_compact!(a; alg = alg isa SDD ? SafeDivideAndConquer() : QRIteration())
	s = diagview(S)
	d_old = length(s)
	s, err = _truncate!(s, trunc)
	d = length(s)
	if d == d_old
		return u, s, v, err
	else
		return u[:, 1:d], s, v[1:d, :], err
	end
end

"""
    tsvd(a; trunc=NoTruncation(), alg=SDD())

Non-mutating version of [`tsvd!`](@ref): the input matrix is copied before the
decomposition (see the matrix method for the reconstruction convention and the factor
properties).
"""
tsvd(a::AbstractMatrix; kwargs...) = tsvd!(copy(a); kwargs...)

"""
    tsvd!(a, left, right; trunc=NoTruncation(), alg=SDD())

Truncated SVD of the tensor `a` with dimensions grouped into `left`/`right`; returns
`(u, s, v, err)`.

The flattening `mat(a) = (left) × (right)` is decomposed as `mat(a) = u_mat *
Diagonal(s) * v_mat` (plain multiplication; `v_mat` is the adjoint `V†` of the right
singular vectors, already conjugated — see [`tsvd!`](@ref)), and the factors are
reshaped back to tensors:

* `u` has dimensions `(left..., r)`: an isometry from `left` to the singular index,
  `Σ u[l, r] * conj(u[l', r]) = δ`;
* `v` has dimensions `(r, right...)`: an isometry from `right` to the singular index,
  `Σ v[r, ri] * conj(v[r', ri]) = δ`;
* `err`: the discarded weight; `trunc` selects how many singular values are kept.

Note: the input tensor `a` itself is not modified (the flattened matrix is copied
internally, since the decomposition requires a contiguous workspace); the `!` marks the
variant that reuses that copy as workspace.
"""
function tsvd!(a::AbstractArray{T, N}, left::NTuple{N1, Int}, right::NTuple{N2, Int};
	trunc::TruncationScheme=NoTruncation(), alg::Union{SVD,SDD}=SDD()) where {T <: Number, N, N1, N2}
	b, ushape, vshape = _tomat(a, left, right)
	isa(b, StridedMatrix) || (b = copy(b))
	u, s, v, err = tsvd!(b; trunc=trunc, alg=alg)
	md = length(s)
	return reshape(u, (ushape..., md)), s, reshape(v, (md, vshape...)), err
end

"""
    tsvd(a, left, right; trunc=NoTruncation(), alg=SDD())

Non-mutating version of [`tsvd!`](@ref): the input tensor is copied before the
decomposition.
"""
tsvd(a::AbstractArray, left::NTuple{N1, Int}, right::NTuple{N2, Int}; kwargs...) where {N1, N2} = tsvd!(copy(a), left, right; kwargs...)

"""
    leftorth!(A; alg=QRpos(), atol=0)

Keyword version of `leftorth!` acting on plain matrices, equivalent to `leftorth!(A, alg, atol)`.
"""
leftorth!(A::StridedMatrix; alg::Union{QR,QRpos,SVD,SDD,Polar}=QRpos(), atol::Real=zero(float(real(scalartype(A))))) = leftorth!(A, alg, atol)
"""
    leftorth!(A, left, right; alg=QRpos(), atol=0)

Left-orthogonalize the tensor `A` with dimensions grouped into `left`/`right`, returning `(u, v)`,
where `u` has dimensions `(left..., s)` and `v` has dimensions `(s, right...)`.

Note: the input tensor `A` itself is not modified (the permuted matrix is copied internally, since the
factorization requires a contiguous workspace); the `!` marks the variant that reuses that copy as workspace.
"""
function leftorth!(A::AbstractArray{T, N}, left::NTuple{N1, Int}, right::NTuple{N2, Int};
					alg::Union{QR,QRpos,SVD,SDD,Polar}=QRpos(), atol::Real=zero(float(real(scalartype(A))))) where {T, N, N1, N2}
	A2, dimu, dimv = _tomat(A, left, right)
	isa(A2, StridedMatrix) || (A2 = copy(A2))
	u, v = leftorth!(A2, alg, atol)
	s = size(v, 1)
	return reshape(u, dimu..., s), reshape(v, s, dimv...)
end

"""
    leftorth(A; alg=QRpos(), atol=0)
    leftorth(A, left, right; alg=QRpos(), atol=0)

Non-mutating version of `leftorth!`: the input is copied before orthogonalization.
"""
leftorth(A::AbstractMatrix; kwargs...) = leftorth!(copy(A); kwargs...)
leftorth(A::AbstractArray, left::NTuple{N1, Int}, right::NTuple{N2, Int}; kwargs...) where {N1, N2} = leftorth!(copy(A), left, right; kwargs...)

"""
    rightorth!(A; alg=LQpos(), atol=0)

Keyword version of `rightorth!` acting on plain matrices, equivalent to `rightorth!(A, alg, atol)`.
"""
rightorth!(A::StridedMatrix; alg::Union{LQ,LQpos,SVD,SDD,Polar}=LQpos(), atol::Real=zero(float(real(scalartype(A))))) = rightorth!(A, alg, atol)
"""
    rightorth!(A, left, right; alg=LQpos(), atol=0)

Right-orthogonalize the tensor `A` with dimensions grouped into `left`/`right`, returning `(u, v)`,
where `u` has dimensions `(left..., s)` and `v` has dimensions `(s, right...)`.

Note: the input tensor `A` itself is not modified (the permuted matrix is copied internally, since the
factorization requires a contiguous workspace); the `!` marks the variant that reuses that copy as workspace.
"""
function rightorth!(A::AbstractArray{T, N}, left::NTuple{N1, Int}, right::NTuple{N2, Int};
					alg::Union{LQ,LQpos,SVD,SDD,Polar}=LQpos(), atol::Real=zero(float(real(scalartype(A))))) where {T, N, N1, N2}
	A2, dimu, dimv = _tomat(A, left, right)
	isa(A2, StridedMatrix) || (A2 = copy(A2))
	u, v = rightorth!(A2, alg, atol)
	s = size(v, 1)
	return reshape(u, dimu..., s), reshape(v, s, dimv...)
end

"""
    rightorth(A; alg=LQpos(), atol=0)
    rightorth(A, left, right; alg=LQpos(), atol=0)

Non-mutating version of `rightorth!`: the input is copied before orthogonalization.
"""
rightorth(A::AbstractMatrix; kwargs...) = rightorth!(copy(A); kwargs...)
rightorth(A::AbstractArray, left::NTuple{N1, Int}, right::NTuple{N2, Int}; kwargs...) where {N1, N2} = rightorth!(copy(A), left, right; kwargs...)

function _tomat(a::AbstractArray{T, N}, left::NTuple{N1, Int}, right::NTuple{N2, Int}) where {T<:Number, N, N1, N2}
	(N == N1 + N2) || throw(DimensionMismatch())
	newindex = (left..., right...)
	a1 = permute(a, newindex)
	shape_a = size(a1)
	dimu = ntuple(i->shape_a[i], N1)
	s1 = prod(dimu)
	dimv = ntuple(i->shape_a[N1+i], N2)
	s2 = prod(dimv)
	return reshape(a1, s1, s2), dimu, dimv
end


"""
    renyi_entropy(v::AbstractVector{<:Real}; α::Real=1)

Compute the Rényi entropy of the vector `v`. For `α=1` this reduces to the Shannon entropy `-Σᵢ vᵢ log(vᵢ)`,
and otherwise it is `1/(1-α) * log(Σᵢ vᵢ^α)`.
The entries of `v` must be nonnegative and sum to 1 (e.g. normalized squared singular values).
"""
function renyi_entropy(v::AbstractVector{<:Real}; α::Real=1)
	α = convert(scalartype(v), α)
	a = _check_and_filter(v)
	if α==one(α)
		return -dot(a, log.(a))
	else
		a = a.^(α)
		return (1/(1-α)) * log(sum(a))
	end
end


function _check_and_filter(v::AbstractVector{<:Real}; tol::Real=1.0e-12)
	(abs(sum(v) - 1) <= tol) || throw(ArgumentError("sum of singular values not equal to 1"))
	oo = zero(scalartype(v))
	for item in v
		((item < oo) && (-item > tol)) && throw(ArgumentError("negative singular values"))
	end
	return [item for item in v if abs(item) > tol]
end

"""
    permutation2swaps(perm)

Decompose the permutation `perm` into a sequence of adjacent transpositions: applying
`swap!(x, b)` (exchanging the sites `b` and `b+1` of an MPS/MPO `x`) for each `b` in the
returned list, in order, transforms a tensor network into the permuted one.
"""
function permutation2swaps(perm)
	p = collect(perm)
	@assert isperm(p)
	swaps = Vector{Int}()
	N = length(p)
	for k in 1:(N - 1)
		append!(swaps, (p[k] - 1):-1:k)
		for l in (k + 1):N
			if p[l] < p[k]
				p[l] += 1
			end
		end
		p[k] = k
	end
	return swaps
end
