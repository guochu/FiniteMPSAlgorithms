@testset "timeevo (TDVP1)" begin
	Random.seed!(99)
	L = 6
	p = model_params(L)
	H = mpo_model(p)
	Hd = dense_model(p)
	ψ = randommps(ComplexF64, fill(2, L); D=16)
	normalize!(ψ)

	# --- real-time evolution vs exact ---
	dt = 0.05
	nsteps = 10
	alg = TDVP1(stepsize=-im * dt)   # stepsize = -im*τ: one sweep of exp(-i H τ)
	env = DMRGCache(H, ψ)
	ψ_ref = todense(ψ)
	for t in 1:nsteps
		sweep!(env, alg)
	end
	vψ = todense(env.ket)
	ψ_exact = exp(Matrix(-im * Hd * (dt * nsteps))) * ψ_ref
	# fidelity (both unitary-normalized)
	@test abs(dot(vψ, ψ_exact))^2 / (norm(vψ) * norm(ψ_exact)) > 1 - 1e-8
	# unitarity: norm conservation
	@test norm(env.ket) ≈ 1 atol = 1e-8

	# --- imaginary-time evolution converges to the ground state ---
	# (the non-uniform next-nearest-neighbour model converges more slowly than the old
	# uniform TFIM, so more steps and a looser tolerance are needed)
	ψg = randommps(ComplexF64, fill(2, L); D=8)
	normalize!(ψg)
	envg = DMRGCache(H, ψg)
	alg_im = TDVP1(stepsize=-0.2)   # stepsize = -τ: one sweep of exp(-H τ)
	for _ in 1:150
		sweep!(envg, alg_im)
	end
	nrm = norm(envg.ket)
	E = real(expectation(H, envg.ket)) / nrm^2
        E_exact = eigmin(Hermitian(Hd))
        @test E ≈ E_exact atol = 5e-3
end

@testset "timeevo coverage" begin
	Random.seed!(77)
	p = model_params(4)
	H = mpo_model(p)

	# complex_stepper
	dt = 0.1
	dt1, dt2 = complex_stepper(dt)
	@test dt1 + dt2 ≈ dt
	@test imag(dt1) ≈ -imag(dt2)

	# timeevompo: evolve a single SchurMPOTensor by WI
	Wschur = H[1]
	Wevolved = timeevompo(Wschur, 0.1, WI())
	@test Wevolved isa AbstractSparseMPOTensor
	# timeevompo on full Hamiltonian
	Hevolved = timeevompo(H, 0.05, WII())
	@test Hevolved isa MPOHamiltonian
	# default-algorithm form
	Hevolved2 = timeevompo(H, 0.05)
	@test Hevolved2 isa MPOHamiltonian
	# stepper type hierarchy
	@test WI() isa FirstOrderStepper && WII() isa FirstOrderStepper
	@test ComplexStepper() isa SecondOrderStepper
	@test WI() isa FiniteMPSAlgorithms.MPSAlgorithm && ComplexStepper(stepper=WII()) isa FiniteMPSAlgorithms.MPSAlgorithm
	@test ComplexStepper().stepper isa WII   # default stepper
end

@testset "Lindblad open-system evolution (WI/WII/TDVP) vs ED" begin
	Random.seed!(7)
	L = 4
	d4 = fill(4, L)   # vectorized density matrix: one site carries (po, pi), po slower

	# model H = -Σ h_i X_i - J Σ Z_i Z_{i+1} (real symmetric terms);
	# jumps √γx·X on site 2 and √γm·σ⁻ on site 3
	h = [0.7, -0.4, 0.9, 0.2]; J = 0.6; γx = 0.35; γm = 0.55
	X = Float64[0 1; 1 0]; Z = Float64[1 0; 0 -1]; lm = ComplexF64[0 0; 1 0]
	I2 = Matrix{ComplexF64}(I, 2, 2); I4 = Matrix{ComplexF64}(I, 4, 4)

	# The Lindblad generator 𝓛 becomes an ordinary (Schur-form) MPO Hamiltonian on the
	# vectorized chain: a Hermitian term t·OᵢOⱼ of H contributes -i·t·O^po + i·t·O^pi,
	# and a single-site jump √γ·L contributes the local 4×4 dissipator map
	#   kron(L, conj(L)) - ½(kron(D, I2) + kron(I2, Dᵀ)),  D = L†L.
	upo(O) = kron(O, I2)                 # O acting on the row (po) index
	upi(O) = kron(I2, transpose(O))      # O acting on the column (pi) index
	function lindblad_map(Lm)
		D = Lm' * Lm
		return kron(Lm, conj(Lm)) .- 0.5 .* (kron(D, I2) .+ kron(I2, transpose(D)))
	end
	terms = OpTerm[]
	append!(terms, [OpTerm(+im * h[i], i => upo(X)) for i in 1:L])          # -i·(-hᵢX)·ρ
	append!(terms, [OpTerm(-im * h[i], i => upi(X)) for i in 1:L])          # +i·ρ·(-hᵢXᵀ)
	append!(terms, [OpTerm(+im * J, i => upo(Z), i + 1 => upo(Z)) for i in 1:L-1])
	append!(terms, [OpTerm(-im * J, i => upi(Z), i + 1 => upi(Z)) for i in 1:L-1])
	push!(terms, OpTerm(1.0, 2 => lindblad_map(√γx * X)))
	push!(terms, OpTerm(1.0, 3 => lindblad_map(√γm * lm)))
	𝓛 = MPOHamiltonian(OpSum(d4, terms))

	# initial state: a nontrivial product operator (vectorized), bond dimension 1
	mloc = (I2 .+ 0.3 .* X .- 0.2 .* Z) ./ 2   # 2×2 local factor (tr = 1)
	ρ = CanonicalMPS([reshape(vec(Matrix{ComplexF64}(mloc)), 1, 4, 1) for _ in 1:L])
	v0 = todense(ρ)

	# ED reference: dense Lindblad superoperator on vec(ρ) in the column-stacking order,
	#   𝓛 = -i(I⊗H - Hᵀ⊗I) + Σ_k (L_k ⊗ conj(L_k) - ½(I⊗D_k + D_kᵀ⊗I)),
	# then permuted into the MPS (interleaved) index order
	Hd = zeros(ComplexF64, 2^L, 2^L)
	for i in 1:L
		Hd .-= h[i] .* reshape(kron(ntuple(k -> k == i ? X : I2, L)...), 2^L, 2^L)
	end
	for i in 1:L-1
		Hd .-= J .* reshape(kron(ntuple(k -> (k == i || k == i + 1) ? Z : I2, L)...), 2^L, 2^L)
	end
	n = 2^L
	𝕀n = Matrix{ComplexF64}(I, n, n)
	super = -im * (kron(𝕀n, Hd) - kron(transpose(Hd), 𝕀n))
	for (Lm, s) in [(√γx * X, 2), (√γm * lm, 3)]
		Ls = kron([k == s ? Matrix{ComplexF64}(Lm) : Matrix{ComplexF64}(I, 2, 2) for k in 1:L]...)
		Ds = Ls' * Ls
		super .+= kron(Ls, conj(Ls)) .- 0.5 .* (kron(𝕀n, Ds) .+ kron(transpose(Ds), 𝕀n))
	end
	bits = collect(Iterators.product(fill(0:1, L)...))
	Perm = zeros(ComplexF64, 4^L, 4^L)
	for po in bits, pi in bits
		col = 1 + sum(po[i] * 2^(L - i) for i in 1:L) + n * sum(pi[i] * 2^(L - i) for i in 1:L)
		row = 1 + sum((2 * po[i] + pi[i]) * 4^(L - i) for i in 1:L)
		Perm[row, col] = 1.0
	end
	T = 0.1
	v0_col = Perm' * vec(v0)              # interleaved → column-stacking
	v_exact = Perm * (exp(super * T) * v0_col)
	nrm(v) = v ./ norm(v)
	relerr(v) = norm(nrm(v) - nrm(v_exact))

	# --- WII (two steps) ---
	ψw = copy(ρ)
	for _ in 1:2
		timeevolve!(ψw, 𝓛, T / 2, WII(); trunc=truncdimcutoff(64, 1.0e-12))
	end
	@test relerr(todense(ψw)) < 5e-3

	# --- WI (first order, 10 steps) ---
	ψwi = copy(ρ)
	for _ in 1:10
		timeevolve!(ψwi, 𝓛, T / 10, WI(); trunc=truncdimcutoff(64, 1.0e-12))
	end
	@test relerr(todense(ψwi)) < 5e-2

	# --- TDVP1 on the non-Hermitian generator (stepsize dt: exp(𝓛·dt)) ---
	ψt = changebond!(copy(ρ); D=16)
	env = DMRGCache(𝓛, ψt)
	alg = TDVP1(stepsize=T / 40, ishermitian=false, verbosity=0)
	for _ in 1:40
		sweep!(env, alg)
	end
	@test relerr(todense(env.ket)) < 5e-2
end

@testset "timeevo (TDVP1, density operator)" begin
	Random.seed!(43)
	σx = Float64[0 1; 1 0]
	σy = ComplexF64[0 -im; im 0]
	σz = Float64[1 0; 0 -1]
	I2 = Matrix{ComplexF64}(I, 2, 2)

	heisenberg_terms(L) = begin
		terms = OpSum(fill(2, L))
		for i in 1:L-1
			push!(terms, OpTerm(1.0, i => σx, i + 1 => σx))
			push!(terms, OpTerm(1.0, i => σy, i + 1 => σy))
			push!(terms, OpTerm(1.0, i => σz, i + 1 => σz))
		end
		terms
	end
	denseH(L; field=Float64[]) = begin
		op(ops...) = reshape(kron(ops...), 2^L, 2^L)
		H = zeros(ComplexF64, 2^L, 2^L)
		for i in 1:L-1
			H .+= op(ntuple(k -> (k == i || k == i + 1) ? σx : I2, L)...)
			H .+= op(ntuple(k -> (k == i || k == i + 1) ? σy : I2, L)...)
			H .+= op(ntuple(k -> (k == i || k == i + 1) ? σz : I2, L)...)
		end
		for (i, c) in enumerate(field)
			H .+= c .* op(ntuple(k -> k == i ? σz : I2, L)...)
		end
		H
	end

	# --- L=1: the tangent space is the full operator space, one step is exact.
	#     Cooling by τ is `stepsize = -τ`, so one sweep applies exp(-τ·h₁) ---
	d = 2
	h1 = randn(ComplexF64, d, d); h1 = (h1 + h1') / 2
	h1mpo = MPO([reshape(h1, 1, d, 1, d)])
	dt = 0.01
	ρ2 = randommpo(ComplexF64, [d]; D=4)
	ρ2d = todense(ρ2)
	env = TDVPCache(h1mpo, ρ2)
	sweep!(env, TDVP1(stepsize=-dt))
	@test norm(todense(env.state) - exp(-dt * h1) * ρ2d) / norm(ρ2d) < 1e-8

	# --- L=1 cooling over many steps: ρ(τ) = e^{-τh} ---
	ρI = tompo(Matrix{ComplexF64}(I, d, d), [d])
	env = TDVPCache(h1mpo, ρI)
	τ = 0.0
	for _ in 1:20
		sweep!(env, TDVP1(stepsize=-0.05))
		τ += 0.05
	end
	@test norm(todense(env.state) - exp(-τ * h1)) / norm(exp(-τ * h1)) < 1e-6

	# --- L=3: the density-operator cooling coincides with the vectorized
	#     `superoperator(h, :left)` route (the same left multiplication h·ρ, driven by a
	#     DMRGCache on the doubled chain) ---
	L = 3
	h = MPOHamiltonian(heisenberg_terms(L))
	ρv = randommpo(ComplexF64, fill(2, L); D=4)
	ψv = vectorize(deepcopy(ρv))
	envA = TDVPCache(h, ρv)
	envB = DMRGCache(superoperator(h, :left), ψv)
	for _ in 1:2
		sweep!(envA, TDVP1(stepsize=-0.05, verbosity=0))
		sweep!(envB, TDVP1(stepsize=-0.05, verbosity=0))
		ρA = todense(envA.state)
		ρB = todense(devectorize(envB.ket))
		@test norm(ρA - ρB) / norm(ρA) < 1e-8
	end

	# --- L=3 with the *full* manifold (a generic three-site operator has MPO bond
	#     dimension 4, so `tompo` of a dense operator fills the whole space): the cooling
	#     must reproduce the dense propagator exactly (the projector splitting is exact
	#     for the projected flow, and here nothing is projected away) ---
	Mr = randn(ComplexF64, 2^L, 2^L); Mr = (Mr + Mr') / 2
	Hd3 = denseH(L)
	ρfull = tompo(Mr, fill(2, L))
	@test all(==(4), bonddims(ρfull))
	nt = 5; dt3 = 0.05
	envi = TDVPCache(h, deepcopy(ρfull))
	for _ in 1:nt
		sweep!(envi, TDVP1(stepsize=-dt3, verbosity=0))
	end
	@test norm(todense(envi.state) - exp(-(dt3 * nt) * Hd3) * Mr) / norm(Mr) < 1e-10

	# --- real-time evolution of a mixed state is two-sided, ρ(t) = e^{-i·t·H}ρe^{+i·t·H},
	#     so it goes through the vectorized superoperator difference and a DMRGCache
	#     (again on the full manifold, where the step is exact) ---
	ρrt = deepcopy(ρfull)
	ψrt = vectorize(ρrt)
	envrt = DMRGCache(superoperator(h, :left) - superoperator(h, :right), ψrt)
	nrt = 5; dt4 = 0.05
	for _ in 1:nrt
		sweep!(envrt, TDVP1(stepsize=-im * dt4, verbosity=0))
	end
	ρ_exact = exp(-im * (dt4 * nrt) * Hd3) * Mr * exp(im * (dt4 * nrt) * Hd3)
	@test norm(todense(devectorize(envrt.ket)) - ρ_exact) / norm(ρ_exact) < 1e-8

	# --- L=3 cooling from the identity: monotone energy decrease (with a random field;
	#     for a purely interacting h the projected flow vanishes at the product state) ---
	termsf = heisenberg_terms(L)
	field = randn(L)
	for i in 1:L
		push!(termsf, OpTerm(field[i], i => σz))
	end
	hf = MPOHamiltonian(termsf)
	Hf = denseH(L; field)
	env = TDVPCache(hf, tompo(Matrix{ComplexF64}(I, 2^L, 2^L), fill(2, L)))
	prevE = Inf
	for _ in 1:6
		sweep!(env, TDVP1(stepsize=-0.25))
		ρdN = todense(env.state)
		E = real(tr(Hf * ρdN) / tr(ρdN))
		@test E <= prevE + 1e-10
		prevE = E
	end

	# --- both cache constructors agree ---
	hd = MPO(h)
	ρt1 = randommpo(ComplexF64, fill(2, L); D=2)
	ρt2 = deepcopy(ρt1)
	e1 = TDVPCache(h, ρt1)
	e2 = TDVPCache(hd, ρt2)
	sweep!(e1, TDVP1(stepsize=-0.1))
	sweep!(e2, TDVP1(stepsize=-0.1))
	@test todense(e1.state) == todense(e2.state)

	# --- CanonicalMPS state: the same cache and the same sweeps as DMRGCache ---
	ψ1 = randommps(ComplexF64, fill(2, L); D=4)
	normalize!(ψ1)
	ψ2 = deepcopy(ψ1)
	ev1 = TDVPCache(h, ψ1)
	ev2 = DMRGCache(h, ψ2)
	@test ev1.estorage == ev2.hstorage
	sweep!(ev1, TDVP1(stepsize=-0.1, verbosity=0))
	sweep!(ev2, TDVP1(stepsize=-0.1, verbosity=0))
	@test todense(ev1.state) ≈ todense(ev2.ket)
	@test all(ev1.estorage[i] ≈ ev2.hstorage[i] for i in eachindex(ev1.estorage))
end

@testset "timeevo (TDVP2, two-site)" begin
	Random.seed!(171)
	L = 4
	ds = fill(2, L)
	p = model_params(L)
	H = MPO(mpo_model(p))
	Hd = dense_model(p)
	τ = 0.05

	# --- alg.trunc caps the bond profile: the pair SVD never exceeds it ---
	ψc = randommps(ComplexF64, ds; D=16)
	envc = DMRGCache(H, ψc)
	sweep!(envc, TDVP2(stepsize=-τ, trunc=truncdim(2), verbosity=0))
	@test all(<=(2), bonddims(envc.ket))

	# --- full manifold: every pair manifold is the full two-site space and every
	#     backward step acts on a full-rank center, so the pair updates and their
	#     re-splittings reproduce the exact propagator e^{-τ·H} ---
	ψ0 = randommps(ComplexF64, ds; D=16)
	ψ0d = todense(ψ0)
	env0 = DMRGCache(H, ψ0)
	sweep!(env0, TDVP2(stepsize=-τ, trunc=NoTruncation(), verbosity=0))
	@test norm(todense(env0.ket) - exp(-τ * Hd) * ψ0d) / norm(ψ0d) < 1e-10

	# --- bond growth: a bond-dimension-1 product state has no bond to vary, so the
	#     bonds have to be created by the pair updates themselves ---
	ψp = prodmps(ComplexF64, ds, fill(1, L))
	@test all(==(1), bonddims(ψp))
	ψpd = todense(ψp)                              # the sweeps mutate ψp in place
	dt = 0.05
	nrt = 10
	envp = DMRGCache(H, ψp)
	for _ in 1:nrt
		sweep!(envp, TDVP2(stepsize=-im * dt, trunc=truncdim(4), verbosity=0))
	end
	@test bonddims(envp.ket) == [2, 4, 2]          # grown to the full profile
	vψ = todense(envp.ket)
	ψex = exp(Matrix(-im * Hd * (dt * nrt))) * ψpd
	@test abs(dot(vψ, ψex))^2 / (norm(vψ) * norm(ψex)) > 1 - 1e-4

	# --- imaginary-time cooling from the bond-dimension-1 infinite-temperature state
	#     (the benchmark's small-scale version): no `changebond!` padding, the two-sided
	#     vectorized route grows the bonds and cools towards the Gibbs state. The first
	#     sweep still runs while the bonds grow, and its projectors are not yet the full
	#     tangent space, so a fixed O(dτ) term remains — the same one MPSKit's TDVP2
	#     shows on this model (2.2721e-3 / 1.1356e-3 / 5.6774e-4 at nst = 20/40/80), so
	#     the bounds below are that term, not the roundoff level ---
	β = 1.0
	ψI = vectorize(infinite_temperature_state(ComplexF64, ds))
	@test all(==(1), bonddims(ψI))
	@test !iscanonical(ψI)                # the identity's site tensors are not isometries
	rightorth!(ψI)                        # gauge reset only: state and profile are kept
	@test iscanonical(ψI)
	@test all(==(1), bonddims(ψI))
	@test norm(todense(devectorize(ψI)) - I(2^L) / 2^L) < 1e-12
	G = superoperator(H, :left) + superoperator(H, :right)
	ex = exp(Matrix(-β * Hermitian(Hd)))
	ρed = ex / tr(ex)
	errs = Float64[]
	for nst in (20, 40)
		envI = DMRGCache(G, copy(ψI))
		for _ in 1:nst
			sweep!(envI, TDVP2(stepsize=-β / 2 / nst, trunc=truncdim(16), verbosity=0))
		end
		@test bonddims(envI.ket) == [4, 16, 4]
		ρ = todense(devectorize(envI.ket))
		ρ ./= tr(ρ)
		push!(errs, norm(ρ - ρed) / norm(ρed))
	end
	@test errs[1] < 5e-3
	@test errs[2] < errs[1] / 1.8          # the growth-phase error is O(dτ)

	# --- the density-operator manifold (TDVPCache with a CanonicalMPO state) grows its
	#     bonds the same way, from the unnormalized bond-dimension-1 identity ---
	L3 = 3
	ds3 = fill(2, L3)
	p3 = model_params(L3)
	h3 = MPOHamiltonian(mpo_model(p3))
	Hd3 = dense_model(p3)
	ρg = tompo(Matrix{ComplexF64}(I, 2^L3, 2^L3), ds3)
	@test all(==(1), bonddims(ρg))
	rightorth!(ρg)
	@test iscanonical(ρg)
	envg = TDVPCache(h3, ρg)
	nst3 = 6
	for _ in 1:nst3
		sweep!(envg, TDVP2(stepsize=-β / nst3, trunc=truncdim(4), verbosity=0))
	end
	@test maximum(bonddims(envg.state)) > 1
	ρ3 = todense(envg.state)
	ρ3 ./= tr(ρ3)
	ex3 = exp(Matrix(-β * Hermitian(Hd3)))
	ρed3 = ex3 / tr(ex3)
	@test norm(ρ3 - ρed3) / norm(ρed3) < 1e-4
end
