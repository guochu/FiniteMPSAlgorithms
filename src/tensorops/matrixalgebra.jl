# vendored from TEMPO/src/tensorops/matrixalgebra.jl

abstract type FactorizationAlgorithm end
"""
    OrthogonalFactorizationAlgorithm

Abstract type for orthogonal factorization algorithms; concrete algorithms include `QR`, `QRpos`, `LQ`, `LQpos`, `SVD`, `SDD`, and `Polar`.
The factorizations are computed via the [MatrixAlgebraKit](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl) backend.
"""
abstract type OrthogonalFactorizationAlgorithm <: FactorizationAlgorithm end

struct QRpos <: OrthogonalFactorizationAlgorithm
end
struct QR <: OrthogonalFactorizationAlgorithm
end
struct LQ <: OrthogonalFactorizationAlgorithm
end
struct LQpos <: OrthogonalFactorizationAlgorithm
end
struct SDD <: OrthogonalFactorizationAlgorithm # lapack's default divide and conquer algorithm
end
struct SVD <: OrthogonalFactorizationAlgorithm
end
struct Polar <: OrthogonalFactorizationAlgorithm
end

Base.adjoint(::QRpos) = LQpos()
Base.adjoint(::QR) = LQ()
Base.adjoint(::LQpos) = QRpos()
Base.adjoint(::LQ) = QR()

Base.adjoint(alg::Union{SVD,SDD,Polar}) = alg

const OFA = OrthogonalFactorizationAlgorithm

"""
	leftorth!(A, alg, atol=0)

Left-orthogonalize the matrix `A` into `(Q, R)` such that `A = Q * R` with `Q` having orthonormal columns.
`alg` may be `QR`/`QRpos` (QR factorization, `atol` must be 0), `SVD`/`SDD` (singular value
decomposition, where small singular values below `atol` are truncated), or `Polar` (polar
decomposition). Returns `(Q, R)`.

Note: the input `A` is used as workspace and may be **destroyed/overwritten**; pass a copy if the input must be preserved.
"""
function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{QR,QRpos}, atol::Real=zero(float(real(scalartype(A)))))
	iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
	return left_orth!(A; alg = :qr, positive = alg isa QRpos)
end

function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD}, atol::Real=zero(float(real(scalartype(A)))))
	alg′ = TruncatedAlgorithm(alg isa SVD ? QRIteration() : DivideAndConquer(), trunctol(atol = atol))
	return left_orth!(A, LeftOrthAlgorithm{:svd}(alg′))
end

function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Polar, atol::Real=zero(float(real(scalartype(A)))))
	iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
	return left_orth!(A; alg = :polar)
end

"""
	rightorth!(A, alg, atol=0)

Right-orthogonalize the matrix `A` into `(L, Q)` such that `A = L * Q` with `Q` having orthonormal rows.
`alg` may be `LQ`/`LQpos` (LQ factorization, `atol` must be 0), `SVD`/`SDD` (singular value
decomposition, where small singular values below `atol` are truncated), or `Polar` (polar
decomposition). Returns `(L, Q)`.

Note: the input `A` is used as workspace and may be **destroyed/overwritten**; pass a copy if the input must be preserved.
"""
function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{LQ,LQpos}, atol::Real=zero(float(real(scalartype(A)))))
	iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
	return right_orth!(A; alg = :lq, positive = alg isa LQpos)
end

function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD}, atol::Real=zero(float(real(scalartype(A)))))
	alg′ = TruncatedAlgorithm(alg isa SVD ? QRIteration() : DivideAndConquer(), trunctol(atol = atol))
	return right_orth!(A, RightOrthAlgorithm{:svd}(alg′))
end

function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Polar, atol::Real=zero(float(real(scalartype(A)))))
	iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
	return right_orth!(A; alg = :polar)
end
