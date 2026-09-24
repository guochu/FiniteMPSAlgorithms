# sparse (Schur matrix-of-matrices) MPO: construction, dense conversion, WI/WII, dispatch

@testset "sparse MPO" begin
	Random.seed!(1234)
	σx = Float64[0 1; 1 0]
	σz = Float64[1 0; 0 -1]
	L = 6
	p = model_params(L)

	# Schur-form assembly from terms vs product-MPO assembly
	hs = mpo_model(p)          # OpTerm route
	hd = mpo_model_prod(p)     # prodmpo + `+` route

	@testset "Schur construction and dense conversion" begin
		@test hs[1] isa SchurMPOTensor
		# 6 on-site + 5 bond + 4 next-nearest terms -> 9 private channels + 2;
		# every site carries the full logical shape (boundaries are tompotensors' job)
		@test size(hs[1]) == (11, 11)
		@test size(hs[2]) == (11, 11)
		@test size(hs[end]) == (11, 11)
		@test all(W -> size(W) == (11, 11), hs)
		@test scalartype(hs) == ComplexF64
		@test todense(hs) ≈ todense(hd) atol = 1.0e-12
		@test todense(hs) ≈ dense_model(p) atol = 1.0e-12
		@test_throws ArgumentError OpTerm(-1.0, 1 => σz, 1 => σz)
	end

	@testset "rectangular sparse site tensors" begin
		# a general (non-triangular) sparse site tensor with left ≠ right aux dimension
		data = Matrix{Any}(undef, 2, 3)
		data[1, 1] = σx; data[1, 2] = 1.0; data[1, 3] = 0.0
		data[2, 1] = 0.0; data[2, 2] = σz; data[2, 3] = 2.0
		W = SparseMPOTensor(data)
		@test size(W) == (2, 3)
		@test W[1, 2] ≈ isometry(Float64, 2)
		@test W[2, 3] ≈ 2 * isometry(Float64, 2)
		@test all(iszero, W[1, 3])

		# boundary handling is not SchurMPOTensor's business: edge-shaped logical
		# matrices (m < 2 or n < 2) are rejected
		edata = Matrix{Any}(undef, 3, 1)
		edata[1, 1] = 0.0; edata[2, 1] = σx; edata[3, 1] = 1.0
		@test_throws ArgumentError SchurMPOTensor(edata)
		fdata = Matrix{Any}(undef, 1, 3)
		fdata[1, 1] = 1.0; fdata[1, 2] = σx; fdata[1, 3] = 0.0
		@test_throws ArgumentError SchurMPOTensor(fdata)
		# an all-scalar logical matrix cannot infer phydim
		sdata = fill(0.0, 2, 2)
		@test_throws ArgumentError SchurMPOTensor(sdata)
		@test_throws ArgumentError SparseMPOTensor(sdata)
	end

	@testset "expectation and ground state via sparse MPO" begin
		ψ = randommps(ComplexF64, fill(2, L); D=8)
		@test expectation(ψ, hs, ψ) ≈ expectation(ψ, hd, ψ) rtol = 1.0e-10

		tr = truncdimcutoff(24, 1.0e-12)
		Es, ψs = ground_state(hs, DMRG1(maxiter=10, D=24))
		Ed, ψd = ground_state(hd, DMRG1(maxiter=10, D=24))
		@test Es ≈ Ed atol = 1.0e-8
		@test real(Es) ≈ real(eigmin(Hermitian(dense_model(p)))) atol = 1.0e-8
		# the two runs may differ by a global sign: compare the overlap magnitude
		@test abs(dot(ψs, ψd)) > 1 - 1.0e-6
	end

	@testset "mult vs exact MPO application" begin
		ψ = randommps(ComplexF64, fill(2, L); D=8)
		W = MPOHamiltonian(hs)                     # any dense chain
		ψa = mult(W, ψ, SVDCompression(trunc=NoTruncation()))
		ψe = W * ψ
		lmul!(1 / norm(ψa), ψa)                    # compare directions (mult keeps the norm)
		lmul!(1 / norm(ψe), ψe)
		@test distance(ψa, ψe) < 1.0e-6
	end

	@testset "WI/WII time evolution vs exact" begin
		H = dense_model(p)
		Uexact(dt) = exp(-im * dt * Hermitian(H))
		ψ0 = randommps(ComplexF64, fill(2, L); D=16)   # complex evolution needs a complex state
		lmul!(1 / norm(ψ0), ψ0)
		v0 = todense(ψ0)
		tr = truncdimcutoff(64, 1.0e-12)
		dt = -im * 0.1   # real time step τ = 0.1

		ψ1 = copy(ψ0)
		timeevolve!(ψ1, hs, dt, WII(); trunc=tr)
		@test norm(todense(ψ1) - Uexact(0.1) * v0) < 1.0e-2

		# second-order complex stepper: two half steps
		ψ2 = copy(ψ0)
		timeevolve!(ψ2, hs, dt, ComplexStepper(); trunc=tr)
		@test norm(todense(ψ2) - Uexact(0.1) * v0) < norm(todense(ψ1) - Uexact(0.1) * v0) + 1.0e-8
		@test norm(todense(ψ2) - Uexact(0.1) * v0) < 1.0e-2

		# imaginary time with WI (first order, coarser tolerance)
		ψ3 = copy(ψ0)
		timeevolve!(ψ3, hs, -0.02, WI(); trunc=tr)
		@test norm(todense(ψ3) - exp(-0.02 * Hermitian(H)) * v0) / norm(exp(-0.02 * Hermitian(H)) * v0) < 5.0e-2
	end

	@testset "TDVP1/DMRG1 dispatch on sparse MPO" begin
		ψa = randommps(ComplexF64, fill(2, L); D=8)
		ψb = copy(ψa)
		enva = DMRGCache(hs, ψa)
		envb = DMRGCache(hd, ψb)
		sweep!(enva, TDVP1(stepsize=-im * 0.05))
		sweep!(envb, TDVP1(stepsize=-im * 0.05))
		@test distance(ψa, ψb) < 1.0e-6
	end
end
