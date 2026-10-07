# WI / WII: sparse (Schur-form) MPO time-evolution operators.
# Ported from TEMPO src/mpohamiltonian/schurmpo/w1w2.jl;
# reference: arXiv:1407.1832 "Time-evolving a matrix product state with long-ranged interactions"

abstract type FirstOrderStepper <: MPSAlgorithm end
abstract type SecondOrderStepper <: MPSAlgorithm end

"""
	WI(; tol=Defaults.tol, maxiter=Defaults.maxiter)

First-order W-type (WI) time evolution: the evolved MPO tensor is
`W = [[I + dt·D, δ₂·C], [δ₁·B, A]]` with `δ₁δ₂ = dt`.
"""
@kwdef struct WI <: FirstOrderStepper
	tol::Float64 = Defaults.tol
	maxiter::Int = Defaults.maxiter
end

"""
	WII(; tol=Defaults.tol, maxiter=Defaults.maxiter)

First-order W-type (WII) time evolution: each site tensor contains block submatrices
of the block-matrix exponential, giving higher accuracy per step than [`WI`](@ref).
"""
@kwdef struct WII <: FirstOrderStepper
	tol::Float64 = Defaults.tol
	maxiter::Int = Defaults.maxiter
end

"""
	ComplexStepper(stepper::FirstOrderStepper)

Combine a first-order stepper into a second-order stepper: two steps with
`dt₁ = (1-im)·dt/2` and `dt₂ = (1+im)·dt/2` (real time) compose to second order.
"""
@kwdef struct ComplexStepper{F<:FirstOrderStepper} <: SecondOrderStepper
	stepper::F = WII()
end

"""
	complex_stepper(dt) -> (dt₁, dt₂)

The two half steps of [`ComplexStepper`](@ref): `U₁ = exp(H·dt₁)`, `U₂ = exp(H·dt₂)`,
`U₁U₂` is a second-order stepper.
"""
complex_stepper(dt::Number) = ((1 - im) * dt / 2, (1 + im) * dt / 2)

# the evolved W-form site tensor of a Schur tensor: the two unit levels collapse into
# one channel — the vacuum diagonal carries the evolved on-site term — so the logical
# shape is (a+1, c+1) for an (a+2)×(c+2) Schur tensor:
#   [ D(dt)  C' ]
#   [ B'     A  ]
# the finite-chain boundaries select row 1 / column 1 (`_leftrow`/`_rightcol`).

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
	timeevompo(h::MPOHamiltonian, dt, alg=WII())

Evolve a Schur-form Hamiltonian by one step `dt` with the stepping algorithm `alg`
([`WI`](@ref), [`WII`](@ref), or [`ComplexStepper`](@ref)). The single-tensor methods
return the evolved full-logical-shape 4-index tensor; the Hamiltonian method assembles
them into the dense finite-chain [`MPO`](@ref) (the vacuum row of the first site and the
closing column of the last site; the channel-1 diagonal carries the on-site evolution).
`ComplexStepper` returns the pair of half-step results.
"""
function timeevompo(m::SchurMPOTensor, dt::Number, alg::WI)
	# the evolution introduces the scalar type of dt (e.g. complex time for real-time
	# evolution of a real-valued W)
	T = promote_type(scalartype(m), typeof(dt))
	a, c, d = size(m.A, 1), size(m.A, 3), size(m.A, 2)
	δ₁, δ₂ = _sqrt2(dt)
	O = zeros(T, a + 1, d, c + 1, d)
	O[1, :, 1, :] .= isometry(T, d) .+ dt .* m.D
	O[1, :, 2:c+1, :] .= δ₂ .* m.C
	O[2:a+1, :, 1, :] .= δ₁ .* m.B
	O[2:a+1, :, 2:c+1, :] .= m.A
	return O
end

function timeevompo(m::SchurMPOTensor, dt::Number, alg::WII)
	T = promote_type(scalartype(m), typeof(dt))
	a, c, d = size(m.A, 1), size(m.A, 3), size(m.A, 2)
	δ₁, δ₂ = _sqrt2(dt)
	Ddt = dt .* m.D
	WD = exp(Ddt)
	mo = zero(Ddt)
	# the block exponential acts channel by channel: for every interior (row, column)
	# block pair, the closed 4×4 block propagator is exponentiated and its evolved
	# C (starts), B (completions) and A (propagation) blocks read off
	WA = Array{T,4}(undef, a, d, c, d)
	WB = Array{T,3}(undef, a, d, d)
	WC = Array{T,3}(undef, d, c, d)
	for i in 1:a, j in 1:c
		# the block propagator of MPSKit's WII (`WIIStep`): B only couples to D
		# (`∂_t B = BD + DB`), C only to D, A couples to D and to B/C via √δ —
		# exponentiated exactly with the 4d×4d matrix exponential
		tmp = [Ddt                mo                    mo                mo;
			   δ₂ .* m.C[:, j, :] Ddt                   mo                mo;
			   δ₁ .* m.B[i, :, :] mo                    Ddt               mo;
			   m.A[i, :, j, :]    δ₁ .* m.B[i, :, :]    δ₂ .* m.C[:, j, :] Ddt]
		ex = exp(tmp)[:, 1:d]
		WC[:, j, :] = ex[(d+1):2d, :]
		WB[i, :, :] = ex[(2d+1):3d, :]
		WA[i, :, j, :] = ex[(3d+1):4d, :]
	end
	O = zeros(T, a + 1, d, c + 1, d)
	O[1, :, 1, :] .= WD
	O[1, :, 2:c+1, :] .= WC
	O[2:a+1, :, 1, :] .= WB
	O[2:a+1, :, 2:c+1, :] .= WA
	return O
end

function timeevompo(h::MPOHamiltonian, dt::Number, alg::FirstOrderStepper)
	L = length(h)
	O = [timeevompo(h[i], dt, alg) for i in 1:L]
	# the finite chain keeps the vacuum row of the first site and the closing column of
	# the last site (the channel-1 diagonal carries the on-site evolution)
	O[1] = O[1][1:1, :, :, :]
	O[L] = O[L][:, :, 1:1, :]
	return MPO(O)
end

function timeevompo(h::Union{SchurMPOTensor,MPOHamiltonian}, dt::Number, alg::ComplexStepper)
	dt1, dt2 = complex_stepper(dt)
	return timeevompo(h, dt1, alg.stepper), timeevompo(h, dt2, alg.stepper)
end

# two-argument convenience (defaults to WII); a fallback on alg::MPSAlgorithm would be
# ambiguous with the ComplexStepper method above
timeevompo(h::MPOHamiltonian, dt::Number) = timeevompo(h, dt, WII())

# ---------- applying the evolved MPO to a state ----------

# the truncating application of an evolved MPO W to a state is `mult(W, ψ,
# SVDCompression(trunc))` (with the in-place `mult!` variant); there is no dedicated
# `apply` — the previous one silently re-normalized the state, which is wrong for
# non-unitary (e.g. Lindblad) evolution operators.

"""
	timeevo(ψ::CanonicalMPS, h::MPOHamiltonian, dt, alg::FirstOrderStepper;
			trunc=Defaults.alg_trunc()) -> ψ'
	timeevo(ψ, h, dt, alg::ComplexStepper; trunc) -> ψ'

Non-mutating variant of [`timeevo!`](@ref): one W-matrix time step — build the evolved
MPO with [`timeevompo`](@ref) and apply it to `ψ` with the truncation `trunc` (via
[`mult`](@ref)) — returned as a new `CanonicalMPS`, leaving `ψ` untouched. The
`ComplexStepper` composes the two half steps for second-order accuracy.
"""
timeevo(ψ::CanonicalMPS, h::MPOHamiltonian, dt::Number, alg::FirstOrderStepper;
		trunc::TruncationScheme=Defaults.alg_trunc()) =
	mult(timeevompo(h, dt, alg), ψ, SVDCompression(trunc=trunc))

function timeevo(ψ::CanonicalMPS, h::MPOHamiltonian, dt::Number, alg::ComplexStepper;
				 trunc::TruncationScheme=Defaults.alg_trunc())
	dt1, dt2 = complex_stepper(dt)
	φ = mult(timeevompo(h, dt1, alg.stepper), ψ, SVDCompression(trunc=trunc))
	return mult(timeevompo(h, dt2, alg.stepper), φ, SVDCompression(trunc=trunc))
end

"""
	timeevo!(ψ::CanonicalMPS, h::MPOHamiltonian, dt, alg=WII(); trunc=Defaults.alg_trunc())
	timeevo!(ψ, h, dt, ComplexStepper(); trunc)

In-place one W-matrix time step: `copy!`s the result of the non-mutating [`timeevo`](@ref)
into `ψ` (see there for the algorithm; `ComplexStepper` composes the two half steps for
second-order accuracy).
"""
function timeevo!(ψ::CanonicalMPS, h::MPOHamiltonian, dt::Number,
				  alg::MPSAlgorithm=WII(); trunc::TruncationScheme=Defaults.alg_trunc())
	φ = timeevo(ψ, h, dt, alg; trunc)
	# an element-type promotion (e.g. a real state evolved in complex time) cannot be
	# copied back into ψ: return the promoted state directly
	scalartype(φ) == scalartype(ψ) && return copy!(ψ, φ)
	return φ
end
