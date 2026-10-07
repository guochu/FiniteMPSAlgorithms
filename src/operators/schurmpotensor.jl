# sparse MPO site tensors: matrix-of-matrices form with dense block storage, ported
# from InfiniteMPSAlgorithms (MPSKit's JordanMPOTensor). The SchurMPOTensor is the
# upper-triangular (Jordan/Schur) form stored as four dense blocks A/B/C/D, the two
# identity corners being implicit; the logical block matrix may be rectangular
# (left ≠ right channel spaces). Blocks are plain d×d matrices — there is no
# proportional-to-identity scalar compression.

# ---------- SchurMPOTensor (ported from InfiniteMPSAlgorithms) ----------
#
# Upper-triangular block-matrix representation of an MPO (symmetry-free, plain
# Array version). The physical index order matches the rest of the package
# (TEMPO convention `W[wl, u, wr, d]`, bond indices in slots 1 and 3):
#
# ```math
# \begin{pmatrix}
# 1 & C & D \\
# 0 & A & B \\
# 0 & 0 & 1
# \end{pmatrix}
# ```
#
# Unlike the InfiniteMPSAlgorithms original, the logical block matrix may be
# rectangular (`m × n`, `m, n >= 2`): the left and right channel spaces may differ
# (the square case has `m == n`). Block storage (with `a = m - 2` interior rows and
# `c = n - 2` interior columns, unit levels first/last):
# - `A::Array{T,4}`: `(a, d, c, d)`, middle blocks;
# - `B::Array{T,3}`: `(a, d, d)`, middle → last column;
# - `C::Array{T,3}`: `(d, c, d)`, first row → middle;
# - `D::Array{T,2}`: `(d, d)`, first row / last column (the complete on-site term);
# - `[1,1]` and `[m,n]` are implied physical identities (the `BraidingTensor`
#   in MPSKit). The local operators must be square (`d_out == d_in`), since the
#   identity channels need `I(d, d)`.

"""
	SchurMPOTensor{T}

Schur (upper-triangular) block-matrix tensor of an MPO (symmetry-free version); see
the module comment. Supports block-matrix style access `W[i, j]` returning the `(d, d)`
local operator. The logical block matrix may be rectangular — the left and right
channel spaces (`size(W, 1)` / `size(W, 2)`) may differ.
"""
struct SchurMPOTensor{T<:Number}
	A::Array{T,4}
	B::Array{T,3}
	C::Array{T,3}
	D::Array{T,2}
end

Base.copy(W::SchurMPOTensor) = SchurMPOTensor(copy(W.A), copy(W.B), copy(W.C), copy(W.D))
scalartype(::Type{SchurMPOTensor{T}}) where {T} = T
function Base.complex(W::SchurMPOTensor{T}) where {T}
	T <: Complex && return W
	return SchurMPOTensor(complex.(W.A), complex.(W.B), complex.(W.C), complex.(W.D))
end

# the number of virtual levels per side (= interior channels + 2 unit levels), aligned
# with the dense MPOTensor naming; `space_l == space_r` only in the square case
space_l(W::SchurMPOTensor) = size(W.A, 1) + 2
space_r(W::SchurMPOTensor) = size(W.A, 3) + 2

# the logical block matrix: (space_l × space_r)
Base.size(W::SchurMPOTensor) = (space_l(W), space_r(W))
Base.size(W::SchurMPOTensor, i::Int) = i == 1 ? space_l(W) :
	i == 2 ? space_r(W) : throw(BoundsError(W, i))
phydim(W::SchurMPOTensor) = size(W.A, 2)
ophydim(W::SchurMPOTensor) = size(W.A, 2)
iphydim(W::SchurMPOTensor) = size(W.A, 4)

# ---- block-matrix style access: W[i, j] → (d, d) local operator ----

function Base.getindex(W::SchurMPOTensor{T}, i::Int, j::Int) where {T}
	m, n = space_l(W), space_r(W)
	d = size(W.A, 2)
	if (i == 1 && j == 1) || (i == m && j == n)
		return Matrix{T}(I, d, d)
	elseif i == 1 && j == n
		return W.D
	elseif i == 1
		return W.C[:, j - 1, :]
	elseif j == n
		return W.B[i - 1, :, :]
	elseif 1 < i < m && 1 < j < n
		return W.A[i - 1, :, j - 1, :]
	end
	return zeros(T, d, d)
end

function Base.setindex!(W::SchurMPOTensor{T}, O, i::Int, j::Int) where {T}
	d = size(W.A, 2)
	Om = O isa Number ? Matrix{T}(O * I, d, d) : O
	(size(Om, 1) == size(Om, 2) == d) ||
		throw(DimensionMismatch("local operator must be $(d)×$(d)"))
	m, n = space_l(W), space_r(W)
	if (i == 1 && j == 1) || (i == m && j == n)
		# unit diagonal blocks: overwriting is not allowed (identity preserved)
		return W
	elseif i == 1 && j == n
		W.D .= Om
	elseif i == 1
		W.C[:, j - 1, :] .= Om
	elseif j == n
		W.B[i - 1, :, :] .= Om
	elseif 1 < i < m && 1 < j < n
		(i > j) && throw(ArgumentError("not allowed to set the lower-triangular Schur block ($i, $j)"))
		W.A[i - 1, :, j - 1, :] .= Om
	else
		throw(ArgumentError("lower-triangular Schur block ($i, $j) is identically zero and cannot be assigned"))
	end
	return W
end

# ---- constructors ----

function _mpoham_scalar_type(W::AbstractMatrix)
	T = Union{}
	for v in W
		v isa Missing && continue
		v isa AbstractMatrix || v isa Number || throw(ArgumentError("entries must be scalars or matrices"))
		T = promote_type(T, scalartype(v))
	end
	return T === Union{} ? Float64 : T
end

"""
	SchurMPOTensor(W::AbstractMatrix) -> SchurMPOTensor

Construct from an `n × n` operator matrix: `W[i, j]` is the `(d, d)` local
operator from row `i` (left level) to column `j` (right level); entries may be
`Missing`, `Number`s, or matrices. `[1,1]` and `[end,end]` are implied
identities (entries should be `1` or `Missing`).
"""
SchurMPOTensor(W::AbstractMatrix) = SchurMPOTensor{_mpoham_scalar_type(W)}(W)
SchurMPOTensor{T}(W::AbstractMatrix) where {T<:Number} = _schurmpotensor(W, T, nothing)
# lattice-aware variant: the site's physical dimension comes from the lattice, so a site
# whose block matrix is all-scalar — no operator support at all — is a valid idle-identity
# tensor
SchurMPOTensor{T}(W::AbstractMatrix, d::Int) where {T<:Number} = _schurmpotensor(W, T, d)

function _schurmpotensor(W::AbstractMatrix, ::Type{T}, d::Union{Nothing, Int}) where {T<:Number}
	m, n = size(W)
	(m >= 2 && n >= 2) || throw(ArgumentError(
		"W needs the full logical shape (m, n >= 2, unit levels first/last), got ($m, $n) — chain boundaries are handled by tompotensors"))
	if isnothing(d)
		for v in W
			v isa AbstractMatrix && (d = size(v, 1); break)
		end
	end
	isnothing(d) && throw(ArgumentError("no (d, d) local operator entry found in W " *
										"(all-scalar input; pass the site's physical dimension explicitly)"))
	# validate the unit corners
	for (i, j) in ((1, 1), (m, n))
		v = W[i, j]
		(v isa Missing || v isa Number && isone(v)) ||
			throw(ArgumentError("W[$i, $j] must be 1 or Missing (unit levels imply identity)"))
	end
	J = SchurMPOTensor{T}(zeros(T, m - 2, d, n - 2, d), zeros(T, m - 2, d, d),
						  zeros(T, d, n - 2, d), zeros(T, d, d))
	for i in 1:m, j in 1:n
		v = W[i, j]
		v isa Missing && continue
		iszero(v) && continue
		v isa Number ? (J[i, j] = Matrix{T}(v * I, d, d)) : (J[i, j] = v)
	end
	return J
end

"""
	tompotensor(W::SchurMPOTensor) -> Array{T,4}

Densify into the package's 4-index MPO tensor `(wl, u, wr, d)`: the identity
channels live at level `1` and level `space_r`, i.e.
`Wd[1,:,1,:] = Wd[end,:,end,:] = I`.
"""
function tompotensor(W::SchurMPOTensor)
	T = scalartype(W)
	d = size(W.A, 2)
	m, n = space_l(W), space_r(W)
	Wd = zeros(T, m, d, n, d)
	Wd[1, :, 1, :] .= Matrix{T}(I, d, d)
	Wd[m, :, n, :] .= Matrix{T}(I, d, d)
	Wd[1, :, n, :] .= W.D
	for j in 2:(n - 1)
		Wd[1, :, j, :] .= W.C[:, j - 1, :]
	end
	for i in 2:(m - 1)
		Wd[i, :, n, :] .= W.B[i - 1, :, :]
		for j in 2:(n - 1)
			Wd[i, :, j, :] .= W.A[i - 1, :, j - 1, :]
		end
	end
	return Wd
end

"""
	Base.:+(W₁::SchurMPOTensor, W₂::SchurMPOTensor) -> SchurMPOTensor

The block-native sum, aligned with the dense MPO sum: the auxiliary (channel)
dimensions of the interior `A` blocks and the boundary `B`/`C` blocks are **directly
summed** (block-diagonal concatenation — the summands need not share auxiliary
dimensions), and only the `D` corners (the complete on-site terms) add element-wise.
The physical dimensions must match.
"""
function Base.:+(W₁::SchurMPOTensor, W₂::SchurMPOTensor)
	d₁, d₂ = size(W₁.A, 2), size(W₂.A, 2)
	(d₁ == d₂) || throw(ArgumentError("physical dimensions do not match: $d₁ != $d₂"))
	return SchurMPOTensor(cat(W₁.A, W₂.A; dims=(1, 3)), cat(W₁.B, W₂.B; dims=1),
						  cat(W₁.C, W₂.C; dims=2), W₁.D + W₂.D)
end

function LinearAlgebra.lmul!(f::Number, W::SchurMPOTensor)
	# the whole operator scales: every block, identity corners included
	lmul!(f, W.A)
	lmul!(f, W.B)
	lmul!(f, W.C)
	lmul!(f, W.D)
	return W
end
