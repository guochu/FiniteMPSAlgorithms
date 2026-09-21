# WI / WII: sparse (Schur-form) MPO time-evolution operators.
# Ported from TEMPO src/mpohamiltonian/schurmpo/w1w2.jl;
# reference: arXiv:1407.1832 "Time-evolving a matrix product state with long-ranged interactions"

abstract type TimeEvoMPOAlgorithm end
abstract type FirstOrderStepper <: TimeEvoMPOAlgorithm end
abstract type SecondOrderStepper <: TimeEvoMPOAlgorithm end

"""
	WI(; tol=Defaults.tol, maxiter=Defaults.maxiter)

First-order W-type (WI) time evolution: the evolved MPO tensor is
`W = [[I + dt·D, δ₂·C], [δ₁·B, A]]` with `δ₁δ₂ = dt`.
"""
struct WI <: FirstOrderStepper
	tol::Float64
	maxiter::Int
end
WI(; tol::Real=Defaults.tol, maxiter::Int=Defaults.maxiter) = WI(convert(Float64, tol), maxiter)

"""
	WII(; tol=Defaults.tol, maxiter=Defaults.maxiter)

First-order W-type (WII) time evolution: each site tensor contains block submatrices
of the block-matrix exponential, giving higher accuracy per step than [`WI`](@ref).
"""
struct WII <: FirstOrderStepper
	tol::Float64
	maxiter::Int
end
WII(; tol::Real=Defaults.tol, maxiter::Int=Defaults.maxiter) = WII(convert(Float64, tol), maxiter)

"""
	ComplexStepper(stepper::FirstOrderStepper)

Combine a first-order stepper into a second-order stepper: two steps with
`dt₁ = (1-im)·dt/2` and `dt₂ = (1+im)·dt/2` (real time) compose to second order.
"""
struct ComplexStepper{F<:FirstOrderStepper} <: SecondOrderStepper
	stepper::F
end

"""
	complex_stepper(dt) -> (dt₁, dt₂)

The two half steps of [`ComplexStepper`](@ref): `U₁ = exp(H·dt₁)`, `U₂ = exp(H·dt₂)`,
`U₁U₂` is a second-order stepper.
"""
complex_stepper(dt::Number) = ((1 - im) * dt / 2, (1 + im) * dt / 2)

# Schur block extraction: A = interior, B = last column interior (completions),
# C = first row interior (starts), D = top-right corner (on-site terms)
get_A(x::SchurMPOTensor) = [x[i, j] for i in 2:size(x, 1)-1, j in 2:size(x, 2)-1]
get_B(x::SchurMPOTensor) = [x[i, size(x, 2)] for i in 2:size(x, 1)-1]
get_C(x::SchurMPOTensor) = [x[1, j] for j in 2:size(x, 2)-1]
get_D(x::SchurMPOTensor) = x[1, size(x, 2)]

function _SiteW_impl(WA, WB, WC, WD)
	s1, s2 = size(WA)
	r = Matrix{Any}(undef, s1 + 1, s2 + 1)
	r[1, 1] = WD
	for l in 2:s2+1
		r[1, l] = WC[l-1]
	end
	for l in 2:s1+1
		r[l, 1] = WB[l-1]
	end
	r[2:end, 2:end] .= WA
	return SparseMPOTensor(r)
end

function _sqrt2(dt::Complex)
	r = sqrt(dt)
	return r, r
end

function _sqrt2(dt::Real)
	if dt >= zero(dt)
		r = sqrt(dt)
		return r, r
	else
		r = sqrt(-dt)
		return r, -r
	end
end

"""
	timeevompo(m::SchurMPOTensor, dt, alg)
	timeevompo(h::SparseMPOHamiltonian, dt, alg=WII())

Evolve a Schur-form Hamiltonian by one step `dt` with the stepping algorithm `alg`
([`WI`](@ref), [`WII`](@ref), or [`ComplexStepper`](@ref)), returning the evolved sparse
MPO (a `SparseMPOHamiltonian{<:SparseMPOTensor}`; `ComplexStepper` returns the pair of
half-step results).
"""
function timeevompo(m::SchurMPOTensor, dt::Number, alg::WI)
	WA = get_A(m)
	δ₁, δ₂ = _sqrt2(dt)
	WB = get_B(m) .* δ₁
	WC = get_C(m) .* δ₂
	D = get_D(m)
	WD = isometry(scalartype(D), size(D, 1)) + dt * D
	return _SiteW_impl(WA, WB, WC, WD)
end

function timeevompo(m::SchurMPOTensor, dt::Number, alg::WII)
	A, B, C, D = get_A(m), get_B(m), get_C(m), get_D(m)
	Ddt = dt * D
	WD = exp(Ddt)
	d = phydim(m)
	mo = zero(Ddt)
	nA_rows, nA_cols = size(A)
	nB, nC = length(B), length(C)
	δ₁, δ₂ = _sqrt2(dt)

	T = typeof(Ddt)
	WA = Array{T,2}(undef, nA_rows, nA_cols)
	WB = Array{T,1}(undef, nB)
	WC = Array{T,1}(undef, nC)

	# rectangular edge tensors (first row / last column of the chain) have empty A/B or
	# A/C blocks; the missing entries enter the block matrix as zeros and the undefined
	# output slots are simply not stored. `max(..., 1)` keeps the loop running when one
	# of the index sets is empty (e.g. the C evolution on the first site).
	for a in 1:max(nB, nA_rows, 1), b in 1:max(nC, nA_cols, 1)
		Ab = (a <= nA_rows && b <= nA_cols) ? A[a, b] : mo
		Bb = (a <= nB) ? B[a] : mo
		Cb = (b <= nC) ? C[b] : mo
		tmp = [Ddt mo mo mo; δ₂*Cb Ddt mo mo; δ₁*Bb mo Ddt mo; Ab δ₁*Bb δ₂*Cb Ddt]
		ex = exp(tmp)
		ex = ex[:, 1:d]
		(b <= nC) && (WC[b] = ex[(d+1):2d, :])
		(a <= nB) && (WB[a] = ex[(2d+1):3d, :])
		(a <= nA_rows && b <= nA_cols) && (WA[a, b] = ex[(3d+1):4d, :])
	end
	return _SiteW_impl(WA, WB, WC, WD)
end

function timeevompo(h::MPOHamiltonian{<:SchurMPOTensor}, dt::Number, alg::FirstOrderStepper)
	return MPOHamiltonian([timeevompo(h[i], dt, alg) for i in 1:length(h)])
end

function timeevompo(h::Union{SchurMPOTensor,MPOHamiltonian{<:SchurMPOTensor}}, dt::Number, alg::ComplexStepper)
	dt1, dt2 = complex_stepper(dt)
	return timeevompo(h, dt1, alg.stepper), timeevompo(h, dt2, alg.stepper)
end

timeevompo(h::MPOHamiltonian{<:SchurMPOTensor}, dt::Number, alg::TimeEvoMPOAlgorithm=WII()) = timeevompo(h, dt, alg)

# ---------- applying the evolved MPO to a state ----------

# the truncating application of an evolved MPO W to a state is `mult(W, ψ,
# SVDCompression(trunc))` (with the in-place `mult!` variant); there is no dedicated
# `apply` — the previous one silently re-normalized the state, which is wrong for
# non-unitary (e.g. Lindblad) evolution operators.

"""
	timeevolve!(ψ::CanonicalMPS, h::MPOHamiltonian, dt, alg=WII(); trunc=DefaultTruncation)
	timeevolve!(ψ, h, dt, ComplexStepper(WII()); trunc)

One W-matrix time step: build the evolved MPO with [`timeevompo`](@ref) and apply it to
`ψ` with the truncation `trunc` (via [`mult`](@ref)). `ComplexStepper` composes the two
half steps for second-order accuracy.
"""
function timeevolve!(ψ::CanonicalMPS, h::MPOHamiltonian{<:SchurMPOTensor}, dt::Number,
					 alg::TimeEvoMPOAlgorithm=WII(); trunc::TruncationScheme=DefaultTruncation)
	if alg isa ComplexStepper
		dt1, dt2 = complex_stepper(dt)
		W1, W2 = timeevompo(h, dt1, alg.stepper), timeevompo(h, dt2, alg.stepper)
		copy!(ψ, mult(W1, ψ, SVDCompression(trunc)))
		return copy!(ψ, mult(W2, ψ, SVDCompression(trunc)))
	end
	W = timeevompo(h, dt, alg)
	return copy!(ψ, mult(W, ψ, SVDCompression(trunc)))
end
