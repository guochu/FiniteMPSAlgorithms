# Verify the accuracy of TDVP1.
#
# Only reference: the exact unitary propagator U(T) = exp(-i H T) applied to |ψ0>.
#   err_ED(dt) = ‖TDVP(T,dt) - U(T)|ψ0>‖ / ‖U(T)|ψ0>‖
# No TDVP self-reference is used.
#
# For insufficient D, err_ED = floor + c * dt^2 + O(dt^4), where the dt-independent
# manifold-projection floor is estimated as err_ED at the smallest dt. The quantity
# err_ED - floor therefore isolates the time-discretization error.
#
# Findings (non-uniform TFIM, random entangled initial states):
#   * Complete bond manifold (D large enough): err_ED sits at machine precision and
#     is essentially independent of dt — TDVP is EXACT (the MPS manifold contains the
#     true evolution path; each ac/c exponentiation + QR/LQ gauge transport implements
#     exp(-i H dt) with no splitting error). Verified up to T=10, L=8.
#   * D insufficient: err_ED - floor converges with order 2 (halving dt quarters the
#     discretization error). This is the regime where TDVP is a genuine approximation.
using LinearAlgebra, Random
using FiniteMPSAlgorithms

include(joinpath(@__DIR__, "..", "test", "helpers.jl"))

const _SX = Float64[0 1; 1 0]
const _SZ = Float64[1 0; 0 -1]

# ---------- standard uniform transverse-field Ising model ----------
function tfim_mpo(L; J=1.0, h=0.9)
	terms = OpSum(fill(2, L))
	for i in 1:L-1
		push!(terms, OpTerm(-J, i => _SX, i+1 => _SX))
	end
	for i in 1:L
		push!(terms, OpTerm(-h, i => _SZ))
	end
	return MPOHamiltonian(terms)
end
function tfim_dense(L; J=1.0, h=0.9)
	I2 = Matrix{Float64}(I, 2, 2)
	op(ops...) = reshape(kron(ops...), 2^L, 2^L)
	H = zeros(2^L, 2^L)
	for i in 1:L-1
		H .-= J .* op(ntuple(k -> (k == i || k == i+1) ? _SX : I2, L)...)
	end
	for i in 1:L
		H .-= h .* op(ntuple(k -> k == i ? _SZ : I2, L)...)
	end
	return H
end

# ---------- non-uniform transverse-field Ising model ----------
# H = -Σ_i J_i σx_i σx_{i+1} - Σ_i h_i σz_i, with site-dependent coefficients
nti_coeffs(L) = (
	J = [1.0 + 0.6 * sin(2.1 * i + 1.3) for i in 1:L-1],
	h = [0.8 + 0.5 * cos(1.7 * i + 0.4) for i in 1:L],
)
function nti_mpo(L, J, h)
	terms = OpSum(fill(2, L))
	for i in 1:L-1
		push!(terms, OpTerm(-J[i], i => _SX, i+1 => _SX))
	end
	for i in 1:L
		push!(terms, OpTerm(-h[i], i => _SZ))
	end
	return MPOHamiltonian(terms)
end
function nti_dense(L, J, h)
	I2 = Matrix{Float64}(I, 2, 2)
	op(ops...) = reshape(kron(ops...), 2^L, 2^L)
	H = zeros(2^L, 2^L)
	for i in 1:L-1
		H .-= J[i] .* op(ntuple(k -> (k == i || k == i+1) ? _SX : I2, L)...)
	end
	for i in 1:L
		H .-= h[i] .* op(ntuple(k -> k == i ? _SZ : I2, L)...)
	end
	return H
end

# ---------- TDVP driver ----------
function evolve(H, ψ0, T, dt, D)
	ψ = changebond!(copy(ψ0); D)
	env = DMRGCache(H, ψ)
	alg = TDVP1(stepsize=dt, verbosity=0)
	for _ in 1:round(Int, T / dt)
		sweep!(env, alg)
	end
	return env.ket
end

function run(H, Hd, ψ0, T, D, dts, tag)
	ψed = exp(Matrix(-im * Hd * T)) * todense(ψ0)   # strict propagator
	nψ = norm(ψed)
	errs = [norm(todense(evolve(H, ψ0, T, dt, D)) - ψed) / nψ for dt in dts]
	floor = minimum(errs)   # manifold-projection floor (dt -> 0 limit)

	println("\n=== ", tag, " (T=", T, ", ‖H‖=", round(opnorm(Hd), digits=2),
		", floor≈", round(floor, sigdigits=3), ") ===")
	println(rpad("dt", 10), rpad("err_ED", 14), rpad("err_ED-floor", 14), "order")
	prev = nothing
	for (dt, e) in zip(dts, errs)
		disc = max(e - floor, 0.0)
		ord = (prev === nothing || disc == 0 || prev == 0) ? "" :
			round(log(prev / disc) / log(2), digits=2)
		println(rpad(dt, 10), rpad(round(e, sigdigits=4), 14),
			rpad(round(disc, sigdigits=3), 14), ord)
		prev = disc
	end
end

# With a complete manifold, a single half-sweep must equal exp(-iH dt/2) exactly.
function half_check(H, Hd, ψ0, dt, D)
	ψ = changebond!(copy(ψ0); D)
	env = DMRGCache(H, ψ)
	FiniteMPSAlgorithms.leftsweep!(env, TDVP1(stepsize=dt, verbosity=0))
	v = todense(env.ket)
	ref = exp(Matrix(-im * Hd * dt / 2)) * todense(ψ)
	println("\n=== half-sweep exactness (D=$D, dt=$dt): discrepancy = ",
		round(norm(v - ref), sigdigits=3), " ===")
end

function main()
	dts = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
	for (L, Dfull) in [(4, 4), (5, 4)]
		Random.seed!(11)
		ψ0 = normalize!(randommps(ComplexF64, fill(2, L); D=8))

		run(tfim_mpo(L), tfim_dense(L), ψ0, 1.0, Dfull, dts,
			"TFIM L=$L D=$Dfull complete")
		run(tfim_mpo(L), tfim_dense(L), ψ0, 1.0, 2, dts,
			"TFIM L=$L D=2 insufficient")
		L == 5 && run(tfim_mpo(L), tfim_dense(L), ψ0, 1.0, 3, dts,
			"TFIM L=$L D=3 insufficient")

		p = model_params(L)
		run(mpo_model(p), dense_model(p), ψ0, 1.0, Dfull, dts,
			"NNN L=$L D=$Dfull complete")
		run(mpo_model(p), dense_model(p), ψ0, 1.0, 2, dts,
			"NNN L=$L D=2 insufficient")
	end

	# non-uniform TFIM, random entangled states, long evolution times
	L = 5
	J, h = nti_coeffs(L)
	Hn, Hdn = nti_mpo(L, J, h), nti_dense(L, J, h)
	Random.seed!(2025)
	ψr = normalize!(randommps(ComplexF64, fill(2, L); D=8))
	half_check(Hn, Hdn, ψr, 0.2, 8)
	run(Hn, Hdn, ψr, 10.0, 8, [0.2, 0.1, 0.05, 0.025, 0.0125],
		"non-uniform TFIM L=5 D=8 complete")
	run(Hn, Hdn, ψr, 10.0, 3, [0.2, 0.1, 0.05, 0.025, 0.0125],
		"non-uniform TFIM L=5 D=3 insufficient")

	# larger chain, complete manifold
	L = 8
	J, h = nti_coeffs(L)
	Random.seed!(99)
	ψ8 = normalize!(randommps(ComplexF64, fill(2, L); D=16))
	run(nti_mpo(L, J, h), nti_dense(L, J, h), ψ8, 4.0, 16, [0.4, 0.2, 0.1, 0.05],
		"non-uniform TFIM L=8 D=16 complete")
end

main()
