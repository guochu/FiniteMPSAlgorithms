# gauging of CanonicalMPS and CanonicalMPO (left/right orthogonalization and canonicalization)
# follows TEMPO adt/orth.jl; SVD paths record the (normalized) bond spectra in `psi.s`
# MPO is a static quantum operator: it is never re-gauged in place; the internal
# `_leftorth!`/`_rightorth!`/`_canonicalize!` machinery below stays generic and is used by
# the algorithms on their own working copies.
# MPO/CanonicalMPO SVD/QR groupings keep the physical double index (p_out, p_in)
# together; after factorization permute(q, (1, 2, 4, 3)) restores the (aL, p_out, aR, p_in) layout

# ---------- per-site scaling convention (total scaling = scaling^L, as in TEMPO) ----------
# only the canonical chains carry a `scaling` field: plain MPO / MPOHamiltonian keep their
# scale in the data and can neither be read nor rescaled through these functions

scaling(x::Union{CanonicalMPS,CanonicalMPO}) = x.scaling[]
setscaling!(x::Union{CanonicalMPS,CanonicalMPO}, s::Real) = (x.scaling[] = s; x)

"""
    LinearAlgebra.normalize!(ψ)

Reset `scaling` to 1. Under the per-site scaling convention the represented state changes
unless `scaling` was already 1; used after the norm has explicitly been pushed into the data.
"""
function LinearAlgebra.normalize!(x::Union{CanonicalMPS,CanonicalMPO})
	setscaling!(x, 1.0)
	return x
end

# ---------- CanonicalMPS ----------

"""
	leftorth!(ψ::CanonicalMPS; alg::Orthogonalize=Orthogonalize(QR()))

Orthogonalize `ψ` into left-canonical form (in place). `SVD` paths record bond spectra in `ψ.s`
and support truncation; `QR` ignores truncation.
"""
leftorth!(ψ::CanonicalMPS; alg::Orthogonalize=Orthogonalize(QR())) =
	_leftorth!(ψ, alg.orth, alg.trunc, alg.normalize, alg.verbosity)

function _leftorth!(ψ::CanonicalMPS, alg::QR, trunc::TruncationScheme, normalize::Bool, verbosity::Int)
	!isa(trunc, NoTruncation) && @warn "truncation has no effect with QR"
	L = length(ψ)
	for i in 1:L-1
		q, r = leftorth!(ψ[i], (1, 2), (3,))
		ψ[i] = q
		_renormalize!(ψ, r, normalize)
		@tensor tmp[1, 3, 4] := r[1, 2] * ψ[i+1][2, 3, 4]
		ψ[i+1] = tmp
	end
	_renormalize!(ψ, ψ[L], normalize)
	_renormalize_coeff!(ψ, normalize)
	return ψ
end

function _leftorth!(ψ::CanonicalMPS, alg::SVD, trunc::TruncationScheme, normalize::Bool, verbosity::Int)
	L = length(ψ)
	maxerr = 0.0
	for i in 1:L-1
		u, s, v, err = tsvd!(ψ[i], (1, 2), (3,); trunc)
		nr = _renormalize!(ψ, s, normalize)
		rerror = sqrt(err * err / (nr * nr + err * err))
		maxerr = max(maxerr, rerror)
		ψ[i] = u
		v2 = Diagonal(s) * v
		@tensor tmp[-1, -2, -3] := v2[-1, 1] * ψ[i+1][1, -2, -3]
		ψ[i+1] = tmp
		ψ.s[i+1] = s
	end
	(verbosity > 0) && println("Max SVD truncation error in leftorth!: ", maxerr)
	_renormalize!(ψ, ψ[L], normalize)
	_renormalize_coeff!(ψ, normalize)
	return ψ
end

"""
	rightorth!(ψ::CanonicalMPS; alg::Orthogonalize=Orthogonalize(SVD(), normalize=false))

Orthogonalize `ψ` into right-canonical form (in place); the Schmidt values of all bonds are
recorded (normalized, with the norm carried by `scaling`).
"""
rightorth!(ψ::CanonicalMPS; alg::Orthogonalize=Orthogonalize(SVD(), normalize=false)) =
	_rightorth!(ψ, alg.orth, alg.trunc, alg.normalize, alg.verbosity)

function _rightorth!(ψ::CanonicalMPS, alg::QR, trunc::TruncationScheme, normalize::Bool, verbosity::Int)
	!isa(trunc, NoTruncation) && @warn "truncation has no effect with QR"
	L = length(ψ)
	for i in L:-1:2
		l, q = rightorth!(ψ[i], (1,), (2, 3))
		ψ[i] = q
		_renormalize!(ψ, l, normalize)
		@tensor tmp[1, 2, 4] := ψ[i-1][1, 2, 3] * l[3, 4]
		ψ[i-1] = tmp
	end
	_renormalize!(ψ, ψ[1], normalize)
	_renormalize_coeff!(ψ, normalize)
	return ψ, 0.0
end

function _rightorth!(ψ::CanonicalMPS, alg::SVD, trunc::TruncationScheme, normalize::Bool, verbosity::Int)
	L = length(ψ)
	maxerr = 0.0
	for i in L:-1:2
		u, s, v, err = tsvd!(ψ[i], (1,), (2, 3); trunc)
		ψ[i] = v
		nr = _renormalize!(ψ, s, normalize)
		rerror = sqrt(err * err / (nr * nr + err * err))
		maxerr = max(maxerr, rerror)
		u2 = u * Diagonal(s)
		@tensor tmp[-1, -2, -3] := ψ[i-1][-1, -2, 1] * u2[1, -3]
		ψ[i-1] = tmp
		ψ.s[i] = s
	end
	(verbosity > 0) && println("Max SVD truncation error in rightorth!: ", maxerr)
	_renormalize!(ψ, ψ[1], normalize)
	_renormalize_coeff!(ψ, normalize)
	return ψ, maxerr
end

"""
	canonicalize!(ψ::CanonicalMPS; alg::Orthogonalize=Orthogonalize(SVD(), DefaultOrthTruncation, false))
	-> (ψ, err)

Transform `ψ` into canonical form (in place): one QR left-orthogonalization sweep (without
truncation) followed by a right-orthogonalization sweep with `alg` (truncating when `alg.trunc`
says so). After the call `ψ` is right-canonical with all Schmidt values initialized. Returns
`ψ` and the maximal truncation error of the right sweep.
"""
canonicalize!(ψ::CanonicalMPS; alg::Orthogonalize=Orthogonalize(SVD(), DefaultOrthTruncation, false)) =
	_canonicalize!(ψ; alg)

"""
	canonicalize(ψ::CanonicalMPS; kwargs...) -> (ψ′, err)

Non-mutating version of [`canonicalize!`](@ref): returns the canonicalized copy and the
maximal truncation error.
"""
canonicalize(ψ::CanonicalMPS; kwargs...) = canonicalize!(copy(ψ); kwargs...)

# internal variant carrying the truncation error; used by the algorithms on their own
# working copies (the public canonicalize! is a thin wrapper)
function _canonicalize!(ψ::CanonicalMPS; alg::Orthogonalize=Orthogonalize(SVD(), DefaultOrthTruncation, false))
	_leftorth!(ψ, QR(), NoTruncation(), alg.normalize, alg.verbosity)
	err = _rightorth!(ψ, alg.orth, alg.trunc, alg.normalize, alg.verbosity)[2]
	_renormalize_coeff!(ψ, alg.normalize)
	return ψ, err
end

# ---------- CanonicalMPO / generic MPO chains ----------

"""
	leftorth!(ρ::CanonicalMPO; alg::Orthogonalize=Orthogonalize(QR()))

Orthogonalize the density-matrix chain into left-canonical form (in place).
"""
function leftorth!(ρ::CanonicalMPO; alg::Orthogonalize=Orthogonalize(QR()))
	_leftorth!(ρ, alg.orth, alg.trunc, alg.normalize, alg.verbosity)
	return ρ
end

function _leftorth!(h::AbstractMPO, alg::QR, trunc::TruncationScheme, normalize::Bool, verbosity::Int)
	!isa(trunc, NoTruncation) && @warn "truncation has no effect with QR"
	L = length(h)
	for i in 1:L-1
		q, r = leftorth!(h[i], (1, 2, 4), (3,))
		h[i] = permute(q, (1, 2, 4, 3))
		_renormalize!(h, r, normalize)
		@tensor tmp[-1, -2, -3, -4] := r[-1, 1] * h[i+1][1, -2, -3, -4]
		h[i+1] = tmp
	end
	_renormalize!(h, h[L], normalize)
	_renormalize_coeff!(h, normalize)
	return h, 0.0
end

function _leftorth!(h::AbstractMPO, alg::SVD, trunc::TruncationScheme, normalize::Bool, verbosity::Int)
	L = length(h)
	maxerr = 0.0
	for i in 1:L-1
		u, s, v, err = tsvd!(h[i], (1, 2, 4), (3,); trunc)
		nr = _renormalize!(h, s, normalize)
		rerror = sqrt(err * err / (nr * nr + err * err))
		maxerr = max(maxerr, rerror)
		h[i] = permute(u, (1, 2, 4, 3))
		v2 = Diagonal(s) * v
		@tensor tmp[-1, -2, -3, -4] := v2[-1, 1] * h[i+1][1, -2, -3, -4]
		h[i+1] = tmp
		h isa CanonicalMPO && (h.s[i+1] = s)
	end
	(verbosity > 0) && println("Max SVD truncation error in leftorth!: ", maxerr)
	_renormalize!(h, h[L], normalize)
	_renormalize_coeff!(h, normalize)
	return h, maxerr
end

"""
	rightorth!(ρ::CanonicalMPO; alg::Orthogonalize=Orthogonalize(SVD(), normalize=false))

Orthogonalize the density-matrix chain into right-canonical form (in place); the Schmidt
values of all bonds are recorded.
"""
function rightorth!(ρ::CanonicalMPO; alg::Orthogonalize=Orthogonalize(SVD(), normalize=false))
	err = _rightorth!(ρ, alg.orth, alg.trunc, alg.normalize, alg.verbosity)[2]
	return ρ
end

function _rightorth!(h::AbstractMPO, alg::QR, trunc::TruncationScheme, normalize::Bool, verbosity::Int)
	!isa(trunc, NoTruncation) && @warn "truncation has no effect with QR"
	L = length(h)
	for i in L:-1:2
		l, q = rightorth!(h[i], (1,), (2, 3, 4))
		h[i] = q
		_renormalize!(h, l, normalize)
		@tensor tmp[-1, -2, -3, -4] := h[i-1][-1, -2, 1, -4] * l[1, -3]
		h[i-1] = tmp
	end
	_renormalize!(h, h[1], normalize)
	_renormalize_coeff!(h, normalize)
	return h, 0.0
end

function _rightorth!(h::AbstractMPO, alg::SVD, trunc::TruncationScheme, normalize::Bool, verbosity::Int)
	L = length(h)
	maxerr = 0.0
	for i in L:-1:2
		u, s, v, err = tsvd!(h[i], (1,), (2, 3, 4); trunc)
		h[i] = v
		nr = _renormalize!(h, s, normalize)
		rerror = sqrt(err * err / (nr * nr + err * err))
		maxerr = max(maxerr, rerror)
		u2 = u * Diagonal(s)
		@tensor tmp[-1, -2, -3, -4] := h[i-1][-1, -2, 1, -4] * u2[1, -3]
		h[i-1] = tmp
		h isa CanonicalMPO && (h.s[i] = s)
	end
	(verbosity > 0) && println("Max SVD truncation error in rightorth!: ", maxerr)
	_renormalize!(h, h[1], normalize)
	_renormalize_coeff!(h, normalize)
	return h, maxerr
end

"""
	canonicalize!(ρ::CanonicalMPO; alg::Orthogonalize=Orthogonalize(SVD(), DefaultOrthTruncation, false))
	-> (ρ, err)

Transform the density-matrix chain into canonical form (in place): QR left sweep + `alg`
right sweep. Returns `ρ` and the maximal truncation error of the right sweep.
"""
canonicalize!(ρ::CanonicalMPO; alg::Orthogonalize=Orthogonalize(SVD(), DefaultOrthTruncation, false)) =
	_canonicalize!(ρ; alg)

canonicalize(ρ::CanonicalMPO; kwargs...) = canonicalize!(copy(ρ); kwargs...)

# internal variant carrying the truncation error; used by the algorithms on their own
# working data (plain MPO chains are gauged only through this internal path, never through
# the public in-place API)
function _canonicalize!(h::AbstractMPO; alg::Orthogonalize=Orthogonalize(SVD(), DefaultOrthTruncation, false))
	_leftorth!(h, QR(), NoTruncation(), alg.normalize, alg.verbosity)
	err = _rightorth!(h, alg.orth, alg.trunc, alg.normalize, alg.verbosity)[2]
	_renormalize_coeff!(h, alg.normalize)
	return h, err
end

isleftcanonical(h::AbstractMPO; atol::Real=1e-12) = all(isleftcanonical(h[i]; atol) for i in eachindex(h))
isrightcanonical(h::AbstractMPO; atol::Real=1e-12) = all(isrightcanonical(h[i]; atol) for i in eachindex(h))

# ---------- canonical-form checks ----------

"""
	iscanonical(ψ::CanonicalMPS; atol=1e-8)

Whether `ψ` is in canonical form: all site tensors right-orthogonal, all Schmidt values
initialized, and on every bond `Diagonal(s.^2) ≈` the left environment.
"""
function iscanonical(ψ::CanonicalMPS; atol::Real=1e-8)
	L = length(ψ)
	for i in 1:L
		isrightcanonical(ψ[i]; atol) || return false
	end
	svectors_uninitialized(ψ) && return false
	h = l_LL(ψ, ψ)
	for i in 1:L-1
		h = _updateleft(h, ψ[i], ψ[i])
		isapprox(h, Diagonal(ψ.s[i+1] .^ 2); atol) || return false
	end
	return true
end

function iscanonical(ρ::CanonicalMPO; atol::Real=1e-8)
	L = length(ρ)
	for i in 1:L
		isrightcanonical(ρ[i]; atol) || return false
	end
	svectors_uninitialized(ρ) && return false
	h = l_LL(ρ, ρ)
	for i in 1:L-1
		h = _updateleft(h, ρ[i], ρ[i])
		isapprox(h, Diagonal(ρ.s[i+1] .^ 2); atol) || return false
	end
	return true
end
