# Thermal-state benchmark on Heisenberg chains: TEBD vs TDVP vs PDMRG
#
# Three routes to the low-temperature equilibrium state rho(beta) ~ exp(-beta H):
#   TEBD  - rho vectorized into a d=4 MPS chain, Strang-ordered imaginary-time
#           gates exp(-dt K) on every bond (K the (Hrho+rhoH)/2 generator);
#   TDVP  - TDVP1 on the density-operator manifold with the left multiplication H·rho
#           (TDVPCache).
#           A purely interacting h has a VANISHING projected flow at the product
#           state rho = I (the tangent space carries no two-site correlations), so
#           the leg is TEBD-warmed to beta0 = BETA0 (untimed) and only the
#           beta0 -> beta segment is timed;
#   PDMRG - positive DMRG (free-energy minimization on the purification ansatz),
#           with the full loss history written to pdmrg_loss_L<L>.csv.
#
# TEBD and TDVP share the same step dt. Run with:
#   julia --project=. bench_thermalstate.jl [L ...]        (default: 10 20 30 40)

using FiniteMPSAlgorithms
using LinearAlgebra
using Printf
using Random

const BETA = 2.0            # target inverse temperature
const BETA0 = 0.5           # TEBD warm-up point for the TDVP leg
const DT = 0.05             # shared step: TEBD Trotter step == TDVP stepsize (=-dt, imaginary time)
const DVEC = 128            # bond cap of the vectorized chain / TDVP seed MPO
const EPS_TRUNC = 1e-10

σx = Float64[0 1; 1 0]
σy = ComplexF64[0 -im; im 0]
σz = Float64[1 0; 0 -1]
I2 = Matrix{ComplexF64}(I, 2, 2)

function heisenberg(L)
	terms = OpSum(fill(2, L))
	for i in 1:L-1
		push!(terms, OpTerm(1.0, i => σx, i + 1 => σx))
		push!(terms, OpTerm(1.0, i => σy, i + 1 => σy))
		push!(terms, OpTerm(1.0, i => σz, i + 1 => σz))
	end
	return MPOHamiltonian(terms)
end

dense_heisenberg(L) = begin
	op(ops...) = reshape(kron(ops...), 2^L, 2^L)
	H = zeros(ComplexF64, 2^L, 2^L)
	for i in 1:L-1
		H .+= op(ntuple(k -> (k == i || k == i + 1) ? σx : I2, L)...)
		H .+= op(ntuple(k -> (k == i || k == i + 1) ? σy : I2, L)...)
		H .+= op(ntuple(k -> (k == i || k == i + 1) ? σz : I2, L)...)
	end
	H
end

# ---------- TEBD pieces (vectorized rho: one site carries (po, pin), po the slower index) ----------
upo(A) = kron(A, I2)               # acts on the po (row) index of one vectorized site
upi(A) = kron(I2, transpose(A))    # acts on the pin (column) index

# (Hrho + rhoH)/2 generator of the two-site bond term, 16x16 on sites (i, i+1)
function bond_cooling_gate(step)
	K = zeros(ComplexF64, 16, 16)
	for S in (σx, σy, σz)
		K .+= 0.5 .* (kron(upo(S), upo(S)) + kron(upi(S), upi(S)))
	end
	return reshape(Matrix(exp(-step * Hermitian(K))), 4, 4, 4, 4)
end

function tebd_step!(ψ, L; dt=DT, trunc=truncdimcutoff(DVEC, EPS_TRUNC))
	g_half = bond_cooling_gate(dt / 2)
	g_full = bond_cooling_gate(dt)
	for i in 1:2:L-1
		apply!(GeneralGate((i, i + 1), g_half), ψ; trunc)
	end
	for i in 2:2:L-1
		apply!(GeneralGate((i, i + 1), g_full), ψ; trunc)
	end
	for i in 1:2:L-1
		apply!(GeneralGate((i, i + 1), g_half), ψ; trunc)
	end
	return ψ
end

function run_tebd(h, L; beta=BETA, dt=DT)
	ψ = vectorize(infinite_temperature_state(ComplexF64, fill(2, L)))
	nsteps = round(Int, beta / dt)
	t = @elapsed for _ in 1:nsteps
		tebd_step!(ψ, L; dt)
	end
	ρ = devectorize(ψ)
	return (time=t, rho=ρ, E=expectationvalue(MPO(h), ρ), D=bonddim(ρ))
end

# ---------- TDVP (TDVP1 on the density operator, left multiplication, TEBD-warmed) ----------
function run_tdvp(h, L; beta=BETA, beta0=BETA0, dt=DT)
	ψ = vectorize(infinite_temperature_state(ComplexF64, fill(2, L)))
	nwarm = round(Int, beta0 / dt)
	for _ in 1:nwarm                     # untimed warm-up (correlation build-up)
		tebd_step!(ψ, L; dt)
	end
	ρ = devectorize(ψ)
	env = TDVPCache(h, ρ)
	alg = TDVP1(stepsize=-dt)
	nsteps = round(Int, (beta - beta0) / dt)
	t = @elapsed for _ in 1:nsteps
		sweep!(env, alg)
	end
	return (time=t, warm=beta0, rho=env.state,
			E=expectationvalue(MPO(h), env.state), D=bonddim(env.state))
end

# ---------- PDMRG (purification free-energy minimization) ----------
function pdmrg_energy(h, rho)
	env = ThermalDMRGCache(h, rho)
	s = rho.center[]
	M = rho.mcenter[]
	E = 0.0
	for τ in 1:size(M, 4)
		x = M[:, :, :, τ]
		E += real(dot(x, FiniteMPSAlgorithms.ac_prime(x, env.H[s], env.hstorage[s], env.hstorage[s+1])))
	end
	return E / tr(rho)
end

function run_pdmrg(h, L; beta=BETA)
	alg = PDMRG(trunc=truncdimcutoff(64, 1e-8), R=64, maxiter=200, tol=1e-6, verbosity=0)
	# random starts can occasionally trip LAPACK (SVD non-convergence): retry on fresh seeds
	for attempt in 1:5
		try
			rho = randompmpa(ComplexF64, fill(2, L); D=64, R=alg.R)
			env = ThermalDMRGCache(h, rho)
			t = @elapsed khist = iterative_compute!(env, alg; β=beta)
			E = pdmrg_energy(h, env.rho)
			return (time=t, khist=khist, E=E, D=alg.trunc.D)
		catch e
			e isa InterruptException && rethrow()
			@printf "  PDMRG attempt %d failed (%s), retrying with a fresh start\n" attempt typeof(e).name.name
			flush(stdout)
			Random.seed!(rand(1:10^9))
		end
	end
	error("PDMRG failed for all retry seeds")
end

# ---------- validation against dense evolution ----------
function validate()
	L = 3
	h = heisenberg(L)
	H = dense_heisenberg(L)
	ρ0 = infinite_temperature_state(ComplexF64, fill(2, L))
	ρd0 = todense(ρ0; order=:msb)
	# one TEBD Strang step vs dense
	ψ = vectorize(copy(ρ0))
	tebd_step!(ψ, L)
	err_tebd = norm(todense(devectorize(ψ); order=:msb) -
					exp(-DT / 2 * Hermitian(H)) * ρd0 * exp(-DT / 2 * Hermitian(H))) /
			   norm(ρd0)
	# one TDVP cooling step vs dense at L=1 (the tangent space is the full
	# single-site operator space, so the projected flow is the exact left action)
	d = 2
	h1 = randn(ComplexF64, d, d); h1 = (h1 + h1') / 2
	h1mpo = MPO([reshape(h1, 1, d, 1, d)])
	ρ1 = randommpo(ComplexF64, [d]; D=4)
	ρ1d = todense(ρ1; order=:msb)
	env = TDVPCache(h1mpo, ρ1)
	sweep!(env, TDVP1(stepsize=-DT))
	exact1 = exp(-DT * h1) * ρ1d
	err_tdvp = norm(todense(env.state; order=:msb) - exact1) / norm(exact1)
	@printf "validation: one-step |Δρ|_TEBD(L=3) = %.2e   |Δρ|_TDVP(L=1) = %.2e\n" err_tebd err_tdvp
	err_tebd < 5e-3 && err_tdvp < 1e-8 || error("validation failed")
	return nothing
end

# ---------- driver ----------
function main(Ls)
	validate()
	rows = Tuple[]
	for L in Ls
		Random.seed!(2024 + L)
		h = heisenberg(L)

		r_tebd = run_tebd(h, L)
		@printf "L = %2d  TEBD  : t = %8.2f s   E = %10.5f   E/L = %8.5f   D = %3d\n" L r_tebd.time real(r_tebd.E) real(r_tebd.E) / L r_tebd.D
		flush(stdout)

		r_tdvp = run_tdvp(h, L)
		@printf "L = %2d  TDVP  : t = %8.2f s   E = %10.5f   E/L = %8.5f   D = %3d   (TEBD-warmed to β0 = %.2f, timed β0 → β = %.2f)\n" L r_tdvp.time real(r_tdvp.E) real(r_tdvp.E) / L r_tdvp.D BETA0 BETA
		flush(stdout)

		r_pdmrg = run_pdmrg(h, L)
		@printf "L = %2d  PDMRG : t = %8.2f s   E = %10.5f   E/L = %8.5f   D = %3d   sweeps = %d\n" L r_pdmrg.time real(r_pdmrg.E) real(r_pdmrg.E) / L r_pdmrg.D length(r_pdmrg.khist)
		flush(stdout)

		# exact reference where ED is feasible
		E_ref = NaN
		if L <= 10
			vals = eigen(Hermitian(dense_heisenberg(L))).values
			w = exp.(-BETA .* vals)
			E_ref = sum(vals .* w) / sum(w)
			@printf "L = %2d  ED ref:            E = %10.5f   E/L = %8.5f\n" L E_ref E_ref / L
		end
		flush(stdout)

		# PDMRG loss history -> CSV
		open(joinpath(@__DIR__, "pdmrg_loss_L$L.csv"), "w") do io
			println(io, "sweep,site,betaF")
			for (k, kvals) in enumerate(r_pdmrg.khist), (s, v) in enumerate(kvals)
				println(io, "$k,$s,$v")
			end
		end

		push!(rows, (L, r_tebd, r_tdvp, r_pdmrg, E_ref))
	end

	# results table + csv
	open(joinpath(@__DIR__, "results_L$(Ls[1]).csv"), "w") do io
		println(io, "L,method,dt,beta,time_s,E,E_over_L,bonddim,pdmrg_sweeps,E_ed_ref")
		for (L, r_tebd, r_tdvp, r_pdmrg, E_ref) in rows
			println(io, "$L,TEBD,$DT,$BETA,$(r_tebd.time),$(real(r_tebd.E)),$(real(r_tebd.E) / L),$(r_tebd.D),,$E_ref")
			println(io, "$L,TDVP,$DT,$BETA,$(r_tdvp.time),$(real(r_tdvp.E)),$(real(r_tdvp.E) / L),$(r_tdvp.D),,$E_ref")
			println(io, "$L,PDMRG,,$BETA,$(r_pdmrg.time),$(real(r_pdmrg.E)),$(real(r_pdmrg.E) / L),$(r_pdmrg.D),$(length(r_pdmrg.khist)),$E_ref")
		end
	end

	println("\n================ summary ================")
	@printf "%4s %6s %10s %12s %10s %6s\n" "L" "method" "time_s" "E" "E/L" "D"
	for (L, r_tebd, r_tdvp, r_pdmrg, E_ref) in rows
		@printf "%4d %6s %10.2f %12.5f %10.5f %6d\n" L "TEBD" r_tebd.time real(r_tebd.E) real(r_tebd.E) / L r_tebd.D
		@printf "%4d %6s %10.2f %12.5f %10.5f %6d\n" L "TDVP" r_tdvp.time real(r_tdvp.E) real(r_tdvp.E) / L r_tdvp.D
		@printf "%4d %6s %10.2f %12.5f %10.5f %6d\n" L "PDMRG" r_pdmrg.time real(r_pdmrg.E) real(r_pdmrg.E) / L r_pdmrg.D
		L <= 10 && @printf "%4d %6s %10s %12.5f %10.5f %6s\n" L "EDref" "" E_ref E_ref / L ""
	end

	# PDMRG loss-decay analysis: how does the sweep loss fall?
	println("\n--------- PDMRG loss decay (last loss per sweep) ---------")
	for (L, r_tebd, r_tdvp, r_pdmrg, E_ref) in rows
		lasts = [kvals[end] for kvals in r_pdmrg.khist]
		n = length(lasts)
		# convergence window: sweeps until the relative sweep-to-sweep change < 1e-3
		conv = n
		for k in 2:n
			d = abs(lasts[k] - lasts[k-1]) / max(abs(lasts[k-1]), 1e-14)
			if d < 1e-3
				conv = k
				break
			end
		end
		# geometric rate over the decay region (mean ratio of successive losses)
		lo = max(2, conv ÷ 2)
		rates = [lasts[k] / lasts[k-1] for k in (lo+1):min(conv, n) if lasts[k-1] > 0]
		rate = isempty(rates) ? NaN : sum(rates) / length(rates)
		@printf "L = %2d : first = %.4e   converged sweep = %d/%d   final = %.6e   mean decay ratio ≈ %.4f/sweep\n" L lasts[1] conv n lasts[n] rate
	end
	println("\nresults written to ", joinpath(@__DIR__, "results_L$(Ls[1]).csv"),
			" and pdmrg_loss_L*.csv")
end

Ls = length(ARGS) > 0 ? [parse(Int, a) for a in ARGS] : [10, 20, 30, 40]
main(Ls)
