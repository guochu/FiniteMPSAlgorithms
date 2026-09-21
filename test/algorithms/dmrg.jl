using FiniteMPSAlgorithms
using Test, LinearAlgebra, Random
using FiniteMPSAlgorithms: SVD, QR, QRpos, LQ, LQpos, SDD, Polar

include(joinpath(@__DIR__, "..", "helpers.jl"))

@testset "dmrg" begin
	Random.seed!(2024)
	L = 8
	p = model_params(L)
	H = mpo_model(p)
	Hd = dense_model(p)
	E_exact = eigmin(Hermitian(Hd))

	# DMRG1, random initial state (bond dimension D)
	E, ψ = ground_state(H, DMRG1(maxiter=50, tol=1e-12, verbosity=0); D=32)
	@test real(E) ≈ E_exact atol = 1e-8
	@test norm(ψ) ≈ 1 atol = 1e-8

	# DMRG1 from a caller-provided initial state
	ψ2 = randommps(ComplexF64, fill(2, L); D=8)
	khist = ground_state!(ψ2, H, DMRG1(maxiter=50, tol=1e-12, verbosity=0))
	@test real(expectation(H, ψ2)) ≈ E_exact atol = 1e-8
	# per-sweep energies decrease towards E_exact
	@test abs(last(last(khist)) - E_exact) < abs(first(last(khist)) - E_exact) + 1e-12
	# the ALS sweeps only iterate; guesses and bond caps are caller-supplied

	# unified sweep interface drives the same iteration
	ψ3 = randommps(ComplexF64, fill(2, L); D=32)
	canonicalize!(ψ3)
	env = DMRGCache(H, ψ3)
	for _ in 1:30
		kvals = sweep!(env, DMRG1(maxiter=1, tol=0.0, verbosity=0))
	end
	@test real(expectation(H, ψ3)) ≈ E_exact atol = 1e-8

	# excited state
	p6 = model_params(6)
	H6 = mpo_model(p6)
	Hd6 = dense_model(p6)
	e = eigen(Hermitian(Hd6))
	E0, ψ0d = e.values[1], e.vectors[:, 1]
	E0gs, ψ0 = ground_state(H6, DMRG1(maxiter=50, tol=1e-12, verbosity=0); D=32)
	E1_exact = e.values[2]   # the first excited level (orthogonal to the exact GS)
	E1, ψ1 = excited_state(H6, DMRG1(maxiter=50, tol=1e-12, verbosity=0), ψ0; D=32)
        @test abs(real(E1) - E1_exact) < 1e-6
        @test abs(dot(todense(ψ1), todense(ψ0))) < 1e-6
end

@testset "dmrg coverage" begin
	Random.seed!(77)
	p6 = model_params(6)
	H6 = mpo_model(p6)
	Hd6 = dense_model(p6)
	e = eigen(Hermitian(Hd6))

	# excited_state! (in-place variant, caller-provided random initial state)
	E0gs, ψ0 = ground_state(H6, DMRG1(maxiter=50, tol=1e-12, verbosity=0); D=32)
	ψex = randommps(ComplexF64, fill(2, 6); D=8)
	khist = excited_state!(ψex, H6, [ψ0], DMRG1(maxiter=50, tol=1e-12, verbosity=0))
	E1_exact = e.values[2]
	@test abs(real(expectation(H6, ψex)) - E1_exact) < 1e-6

	# leftsweep! / rightsweep! (explicit single-direction sweeps)
	H = mpo_model(p6)
	ψs = randommps(ComplexF64, fill(2, 6); D=4)
	canonicalize!(ψs)
	env = DMRGCache(H, ψs)
	kleft = leftsweep!(env, DMRG1(maxiter=1, tol=0.0, verbosity=0))
	@test length(kleft) == 6
	kright = rightsweep!(env, DMRG1(maxiter=1, tol=0.0, verbosity=0))
	@test length(kright) == 6

	# ac_prime / c_prime / Heff
	W = H[1]
	hl = env.hstorage[1]
	hr = env.hstorage[2]
	x = ψs[1]
	y_ac = ac_prime(x, W, hl, hr)
	@test size(y_ac) == (size(hl, 1), phydim(W), size(hr, 1))
	heff = Heff(W, hl, hr)
	@test heff isa FiniteMPSAlgorithms.CentralHeff
	y_heff = ac_prime(x, heff)
	@test y_heff ≈ y_ac atol = 1e-12
	hleft_c = randn(ComplexF64, 3, 2, 4)   # (a, w, b)
	hright_c = randn(ComplexF64, 5, 2, 6)  # (c, w, d)
	xmat = randn(ComplexF64, 4, 6)        # (b, d)
	yc = c_prime(xmat, hleft_c, hright_c)
	@test size(yc) == (3, 5)
	# c_prime vs the direct three-tensor contraction
	yref = @tensor yr[a, c] := hleft_c[a, w, b] * xmat[b, e] * hright_c[c, w, e]
	@test yc ≈ yref atol = 1e-12

	# --- ac_prime / Heff: dense branch vs the direct four-tensor contraction ---
	Random.seed!(3)
	Wd = randn(ComplexF64, 3, 2, 4, 2)     # (wL, p', wR, p)
	hl = randn(ComplexF64, 5, 3, 6)        # (a', wL, l)
	hr = randn(ComplexF64, 7, 4, 8)        # (c', wR, r)
	xc = randn(ComplexF64, 6, 2, 8)        # (l, p, r)
	yd = ac_prime(xc, Wd, hl, hr)
	yref = @tensor yr[a, p2, c] := hl[a, wl, l] * Wd[wl, p2, wr, p] * xc[l, p, r] * hr[c, wr, r]
	@test yd ≈ yref atol = 1e-12
	# the bundled Heff applies identically
	@test ac_prime(xc, Heff(Wd, hl, hr)) ≈ yref atol = 1e-12

	# --- sparse branch vs the dense branch on an equivalent operator ---
	blocks = [Wd[wl, :, wr, :] for wl in 1:3, wr in 1:4]
	Ws = SparseMPOTensor(blocks)
	@test ac_prime(xc, Ws, hl, hr) ≈ yref atol = 1e-12
	heff_s = Heff(Ws, hl, hr)
	@test ac_prime(xc, heff_s) ≈ yref atol = 1e-12
	@test heff_s.W === Ws && heff_s.left === hl && heff_s.right === hr
end
