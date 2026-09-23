using FiniteMPSAlgorithms
using Test, LinearAlgebra, Random
using FiniteMPSAlgorithms: SVD, QR, QRpos, LQ, LQpos, SDD, Polar, DefaultTruncation

include(joinpath(@__DIR__, "..", "helpers.jl"))

@testset "arithmetics" begin
	Random.seed!(7)
	L = 6
	p = model_params(L)
	H = mpo_model(p)
	Hd = dense_model(p)
	ψ = randommps(ComplexF64, fill(2, L); D=10)
	vψ = todense(ψ)
	vHψ = Hd * vψ

	# --- mult, SVD route (no truncation: exact) ---
	out = mult(H, ψ, SVDCompression(trunc=truncdim(256)))
	@test todense(out) ≈ vHψ atol = 1e-8
	# mult with truncation keeps the requested bond dimension and stays canonical
	out2 = mult(H, ψ, SVDCompression(trunc=truncdimcutoff(4, 1e-12)))
	@test bonddim(out2) <= 4
	@test isrightcanonical(out2[1])

	# --- mult, DMRG1 (ALS) route ---
	out3 = mult(H, ψ, DMRG1(maxiter=20, tol=1e-10, D=16))
	@test todense(out3) ≈ vHψ atol = 1e-6 * norm(vHψ)

	# --- mult MPO·MPO ---
	prod_exact = H * H
	@test todense(prod_exact) ≈ Hd * Hd atol = 1e-8
	prod_svd = mult(H, H, SVDCompression(trunc=truncdim(64)))
	@test prod_svd isa CanonicalMPO
	# compare the represented dense operators: a cross-gauge transfer-chain `distance`
	# (raw bond-121 vs canonical bond-64) loses ~1e-6 to cancellation even when the
	# operators agree to machine precision
	@test norm(todense(prod_exact) - todense(prod_svd)) < 1e-6
	prod_als = mult(H, H, DMRG1(maxiter=20, tol=1e-10, D=16))
	@test prod_als isa CanonicalMPO
	@test norm(todense(prod_exact) - todense(prod_als)) < 1e-5

	# --- mult MPO·CanonicalMPO: CanonicalMPO output carrying the operand scaling ---
	ρ = randommpo(ComplexF64, fill(2, L); D=2)
	setscaling!(ρ, 0.7)
	rho_svd = mult(H, ρ, SVDCompression(trunc=truncdim(8)))
	rho_als = mult(H, ρ, DMRG1(maxiter=20, tol=1e-10, D=8))
	@test rho_svd isa CanonicalMPO && rho_als isa CanonicalMPO
	# the exact product (and the wrapped chains) carry the operand scaling, so the
	# represented dense operators compare directly, up to the bond-8 truncation error
	# (exact bond is 22). Note the SVD route's `scaling` field additionally absorbs the
	# data norms during canonicalization (data-normalized gauge): the represented value
	# is the contract, not the raw `scaling` number.
	rho_exact = H * ρ
	δ = norm(todense(rho_exact))
	@test norm(todense(rho_svd) - todense(rho_exact)) / δ < 5e-2
	@test norm(todense(rho_als) - todense(rho_exact)) / δ < 5e-2

	# --- add ---
	s = add([ψ, ψ], SVDCompression(trunc=truncdim(64)))
	@test todense(s) ≈ 2 * vψ atol = 1e-8
	s2 = add([ψ, ψ], DMRG1(maxiter=20, tol=1e-10, D=16))
	@test todense(s2) ≈ 2 * vψ atol = 1e-6

	# --- compress ---
	ψbig = mult(H, ψ, SVDCompression(trunc=truncdim(256)))
	c = compress(ψbig, SVDCompression(trunc=truncdimcutoff(4, 1e-10)))
	@test bonddim(c) <= 4
	c2 = compress(ψbig, DMRG1(maxiter=20, tol=1e-10, D=8))
	@test isrightcanonical(c2[1])

	# in-place ALS compression: the guess object is refined and returned as-is
	guess = svdguess_compress(ψbig, 8)
	c3 = compress!(guess, ψbig, DMRG1(maxiter=20, tol=1e-10, D=8))
	@test c3 === guess
	@test todense(c3) ≈ todense(c2) atol = 1e-8
end

@testset "MPO vs CanonicalMPO separation" begin
	Random.seed!(11)
	L = 4
	ds = fill(2, L)
	W = MPO(randommpo(ComplexF64, ds; D=2).data)
	ρ = randommpo(ComplexF64, ds; D=2)

	# MPO is a static quantum operator: the in-place gauging API is CanonicalMPO-only.
	# (randommpo returns a CanonicalMPO — wrap its data into a plain MPO container)
	@test_throws MethodError canonicalize!(W)
	@test_throws MethodError leftorth!(W)
	@test_throws MethodError rightorth!(W)
	@test_throws MethodError iscanonical(W)
	# scaling accessors and in-place rescaling are CanonicalMPO-only as well
	@test_throws MethodError scaling(W)
	@test_throws MethodError setscaling!(W, 1.0)
	@test_throws MethodError normalize!(W)
	# CanonicalMPO (density matrix) supports the in-place gauging API
	canonicalize!(ρ)
	@test iscanonical(ρ)
	# read-only tensor-isometry checks are available for both chains
	isleftcanonical(W)
	isrightcanonical(ρ)
	# non-mutating canonicalize returns a canonical copy (plus the truncation error)
	W2, _ = canonicalize(ρ)
	@test iscanonical(W2)
end

@testset "scaled inputs (non-unit scaling)" begin
	Random.seed!(21)
	L = 6
	ds = fill(2, L)
	p = model_params(L)
	H = mpo_model(p)
	Hd = dense_model(p)
	ψ = randommps(ComplexF64, ds; D=8)
	φ = randommps(ComplexF64, ds; D=4)
	cs = 2.5
	ψs = ψ * cs                    # per-site scaling; represented norm = cs^L
	@test scaling(ψs) ≈ cs^(1 / L)
	vs = todense(ψs)              # includes scaling^L

	# mult: the represented product must include the input scaling (both routes)
	refm = Hd * vs
	outs = mult(H, ψs, SVDCompression(trunc=truncdim(64)))
	outa = mult(H, ψs, DMRG1(maxiter=20, tol=1e-10, D=16))
	@test scaling(outa) ≈ scaling(ψs) atol = 1e-12
	@test norm(todense(outs) - refm) / norm(refm) < 1e-8
	@test norm(todense(outa) - refm) / norm(refm) < 1e-6

	# add: the sum is expressed in the first input's scaling convention
	s = add([ψs, ψs], DMRG1(maxiter=20, tol=1e-10, D=16))
	@test scaling(s) ≈ scaling(ψs) atol = 1e-12
	@test norm(todense(s) - 2vs) / norm(2vs) < 1e-8

	# compress: the compressed chain represents the scaled input
	c = compress(ψs, DMRG1(maxiter=20, tol=1e-10, D=8))
	@test scaling(c) ≈ scaling(ψs) atol = 1e-12
	@test norm(todense(c) - vs) / norm(vs) < 1e-8

	# hadamard: the result carries the ⊙ convention scaling(ψA)·scaling(ψB)
	φs = φ * 1.7
	refχ = todense(ψs) .* todense(φs)
	χ = hadamard(ψs, φs, DMRG1(maxiter=20, tol=1e-10, D=16))
	@test scaling(χ) ≈ scaling(ψs) * scaling(φs) atol = 1e-12
	@test norm(todense(χ) - refχ) / norm(refχ) < 1e-6

	# linsolve: the solution is expressed in the right-hand side's scaling convention
	I = identitympo(ComplexF64, ds)
	x = linsolve(I, ψs, DMRG1(maxiter=20, tol=1e-10, D=16))
	@test scaling(x) ≈ scaling(ψs) atol = 1e-12
	@test norm(todense(x) - vs) / norm(vs) < 1e-8

	# local expectations include the external scale (OpTerm route)
	σx = Float64[0 1; 1 0]
	I2 = Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)
	Ax = reshape(kron(ntuple(k -> k == 2 ? σx : I2, L)...), 2^L, 2^L)
	ref_loc = real(dot(vs, Ax * vs))
	@test real(expectation(term(2 => σx), ψs)) ≈ ref_loc rtol = 1e-8
	# the normalized value is stable even for a huge external scale
	setscaling!(ψs, 50.0)
	@test real(expectationvalue(term(2 => σx), ψs)) ≈ ref_loc / dot(vs, vs) rtol = 1e-10
end

@testset "long-chain explosion safety" begin
	Random.seed!(2024)
	L = 500
	ds = fill(2, L)
	T = ComplexF64
	# D=1 keeps this cheap; the inputs carry scaling = 50 so the represented norm is
	# 50^L ≈ 1e850 — far beyond Float64. Any stray scaling^L / norm-of-scaled-chain
	# evaluation overflows and shows up as non-finite data or a wrong `scaling` field
	ψ = randommps(T, ds; D=1, normalize=false)
	setscaling!(ψ, 50.0)
	I = identitympo(T, ds)
	datafinite(x) = all(isfinite, reduce(vcat, vec.(x.data)))

	# canonicalize! must not overflow: the gauge factors are data-level and the external
	# scale only moves through the per-site `scaling` field
	ψc = randommps(T, ds; D=1, normalize=false)
	setscaling!(ψc, 50.0)
	canonicalize!(ψc)
	@test datafinite(ψc)
	@test isfinite(scaling(ψc))
	@test iscanonical(ψc)

	# mult (MPS)
	outm = mult(I, ψ)
	@test datafinite(outm)
	@test scaling(outm) ≈ 50.0 atol = 1e-12

	# mult (MPO·MPO, both scaled)
	ρ = randommpo(T, ds; D=1, normalize=false)
	setscaling!(ρ, 50.0)
	outr = mult(ρ, ρ)
	@test datafinite(outr)
	# the SVD route normalizes the data and absorbs the raw product norms into the
	# per-site `scaling` field (data-normalized gauge): it stays finite and the
	# represented value is exact, but it is not the bare 50·50 operand product
	@test isfinite(scaling(outr)) && scaling(outr) > 0

	# add
	s = add([ψ, ψ], DMRG1(maxiter=5, tol=1e-10, D=1))
	@test datafinite(s)
	@test scaling(s) ≈ 50.0 atol = 1e-12

	# compress
	c = compress(ψ, DMRG1(maxiter=5, tol=1e-10, D=1))
	@test datafinite(c)
	@test scaling(c) ≈ 50.0 atol = 1e-12

	# hadamard (pointwise square: the ⊙ convention gives 50·50 = 2500 per site)
	χ = hadamard(ψ, ψ, DMRG1(maxiter=5, tol=1e-10, D=1))
	@test datafinite(χ)
	@test scaling(χ) ≈ 2500.0 atol = 1e-12

	# linsolve
	x = linsolve(I, ψ, DMRG1(maxiter=5, tol=1e-10, D=1))
	@test datafinite(x)
	@test scaling(x) ≈ 50.0 atol = 1e-12
end

@testset "cache coverage" begin
	Random.seed!(55)
	L = 6
	ds = fill(2, L)
	p = model_params(L)
	H = mpo_model(p)
	ψ = randommps(ComplexF64, ds; D=8)

	# DMRGCache
	env = DMRGCache(H, ψ)
	@test env isa DMRGCache
	@test length(env.hstorage) == L + 1
	@test env.H === H
	@test env.ket === ψ
	# the constructor resets the Schmidt values of the working guess
	@test svectors_uninitialized(ψ)

	# OverlapCache: MPS-MPS overlap environment
	ψA = randommps(ComplexF64, ds; D=4)
	ψB = randommps(ComplexF64, ds; D=4)
	oc = OverlapCache(ψA, ψB)
	@test oc isa OverlapCache
	@test length(oc.cstorage) == L + 1
	@test oc.bra === ψA

	# OverlapCache: MPO-MPO overlap environment
	W1 = randommpo(Float64, ds; D=2)
	W2 = randommpo(Float64, ds; D=2)
	oc2 = OverlapCache(W1, W2)
	@test oc2 isa OverlapCache
	@test length(oc2.cstorage) == L + 1

	# ExcitedStateCache: DMRG environments with orthogonality projectors
	ψg = randommps(ComplexF64, ds; D=8)
	proj = [randommps(ComplexF64, ds; D=4)]
	ec = ExcitedStateCache(H, ψg, proj)
	@test ec isa ExcitedStateCache
	@test length(ec.cstorages) == 1

	# recalculate!: copy the new state into the cache's ket and rebuild environments
	ψnew = randommps(ComplexF64, ds; D=8)
	recalculate!(env, ψnew)
	@test env.ket.data == ψnew.data

	# in-place driver entry points (mult! / add! / linsolve!)
	Hd = dense_model(p)
	out_m = mult!(svdguess_mult(H, ψ, 8), H, ψ, DMRG1(maxiter=5, tol=1e-10))
	@test todense(out_m) ≈ Hd * todense(ψ) atol = 1e-4
	out_a = add!(svdguess_add([ψ, ψ], 8), [ψ, ψ], DMRG1(maxiter=5, tol=1e-10))
	@test todense(out_a) ≈ 2 * todense(ψ) atol = 1e-5
	xl = linsolve!(svdguess_compress(ψ, 8), identitympo(ComplexF64, ds), ψ, DMRG1(maxiter=10, tol=1e-10))
	@test abs(dot(xl, ψ)) / (norm(xl) * norm(ψ)) ≈ 1 atol = 1e-6
end

@testset "arithmetics coverage" begin
	Random.seed!(77)
	L = 6
	ds = fill(2, L)
	ψ = randommps(ComplexF64, ds; D=10)

	# Defaults module constants
	@test Defaults.D == 64
	@test Defaults.tol == 1.0e-12
	@test Defaults.maxiter == 100

	# DefaultMultAlg
	@test DefaultMultAlg isa SVDCompression
	@test DefaultMultAlg.trunc == DefaultTruncation

	# truncation scheme types
	tr = truncdimcutoff(8, 1e-10)
	@test tr isa TruncateDimCutoff && tr isa TruncationScheme
	@test truncdim(4) isa TruncateDim

	# svdguess_hadamard: naive-SVD initial guess of the pointwise product
	φ = randommps(ComplexF64, ds; D=4)
	χg = hadamard!(svdguess_hadamard(ψ, φ, 16), ψ, φ, DMRG1(maxiter=5, tol=1e-10))
	ref = todense(ψ) .* todense(φ)
	@test norm(todense(χg) - ref) / norm(ref) < 1e-4

	# svd_add (internal, no longer exported)
	ψ1 = randommps(ComplexF64, ds; D=4)
	ψ2 = randommps(ComplexF64, ds; D=4)
	ψsum, errsum = FiniteMPSAlgorithms.svd_add([ψ1, ψ2], truncdim(8))
	@test todense(ψsum) ≈ todense(ψ1) + todense(ψ2) atol = 1e-8
end

@testset "iterative hadamard" begin
	Random.seed!(123)
	L = 6
	ds = fill(2, L)
	ψ = randommps(ComplexF64, ds; D=4)
	φ = randommps(ComplexF64, ds; D=4)
	ref = todense(ψ) .* todense(φ)

	# exact hadamard needs bond 16; a cap of 32 recovers it essentially exactly
	χ = hadamard(ψ, φ, DMRG1(maxiter=15, tol=1e-10, verbosity=0, D=32))
	@test bonddim(χ) <= 32
	@test norm(todense(χ) - ref) / norm(ref) < 1e-6

	# per-site loss monotonicity inside every sweep (random guess -> actual movement).
	# All losses are in processing-time order. A full sweep processes sites 1:L then
	# L:1, and the local-target norm is non-decreasing throughout
	cache = HadamardCache(ψ, φ, randommps(ComplexF64, ds; D=32))
	khist = iterative_compute!(cache, DMRG1(maxiter=12, tol=1e-11, verbosity=0, D=32))
	χr = cache.bra
	@test all(kv -> length(kv) == 2 * L, khist)
	for kv in khist
		left, right = kv[1:L], kv[L+1:2L]
		@test all(diff(left) .≥ -1e-9)     # leftsweep: processing order, non-decreasing
		@test all(diff(right) .≥ -1e-9)    # rightsweep: processing order, non-decreasing
	end
	# ALS from a random guess still approaches the pointwise product
	@test norm(todense(χr) - ref) / norm(ref) < 1e-4
end

@testset "iterative linsolve" begin
	Random.seed!(901)
	L = 6
	ds = fill(2, L)
	alg = DMRG1(maxiter=20, tol=1e-10, verbosity=0, D=16)

	# trivial system: I·x = y, exact solution is y (up to gauge)
	I_mpo = identitympo(ComplexF64, ds)
	y = randommps(ComplexF64, ds; D=4)
	x = linsolve(I_mpo, y, alg)
	# normalized overlap |⟨x|y⟩|/(‖x‖‖y‖) ≈ 1
	@test abs(dot(x, y)) / (norm(x) * norm(y)) ≈ 1 atol = 1e-7

	# nontrivial system: U·x = y with a unitary time-evolution MPO
	p = model_params(L)
	H = mpo_model(p)
	U = timeevompo(H, 0.15, WII())     # unitary MPOHamiltonian
	x_exact = randommps(ComplexF64, ds; D=4)
	yU = mult(U, x_exact)              # y = U·x_exact
	xsol = linsolve(U, yU, alg)
	# recovered solution: normalized overlap with the exact solution
	@test abs(dot(xsol, x_exact)) / (norm(xsol) * norm(x_exact)) ≈ 1 atol = 1e-5

	# default-algorithm form
	xk = linsolve(I_mpo, y, DMRG1(D=16))
	@test abs(dot(xk, y)) / (norm(xk) * norm(y)) ≈ 1 atol = 1e-7

	# dimension mismatch: a y with wrong physical dimensions
	y5 = randommps(ComplexF64, fill(2, L-1); D=4)
	@test_throws DimensionMismatch linsolve(I_mpo, y5, alg)
end

@testset "arithmetics (DMRG2)" begin
	Random.seed!(4242)
	L = 6
	ds = fill(2, L)
	p = model_params(L)
	H = mpo_model(p)
	Hd = dense_model(p)
	ψ = randommps(ComplexF64, ds; D=10)
	vψ = todense(ψ)
	vHψ = Hd * vψ

	# mult: two-site sweeps grow the bond dimension from a small initial guess
	out = mult(H, ψ, DMRG2(maxiter=10, tol=1e-10, trunc=truncdim(8), verbosity=0))
	@test todense(out) ≈ vHψ atol = 1e-6
	@test bonddim(out) <= 8

	# mult MPO·MPO
	prod2 = mult(H, H, DMRG2(maxiter=10, tol=1e-10, trunc=truncdim(16), verbosity=0))
	@test norm(todense(prod2) - Hd * Hd) < 1e-5

	# mult with the DMRG2 in-place driver (the dense MPO layer; sparse Hamiltonians are
	# expanded by the out-of-place entry points)
	Hd_mpo = MPO(tompotensors(H))
	outm = mult!(svdguess_mult(Hd_mpo, ψ, 4), Hd_mpo, ψ, DMRG2(maxiter=10, tol=1e-10, trunc=truncdim(8)))
	@test todense(outm) ≈ vHψ atol = 1e-5

	# add: the pair updates can express the exact bond-16 sum within the cap
	s2 = add([ψ, ψ], DMRG2(maxiter=10, tol=1e-10, trunc=truncdim(8), verbosity=0))
	@test todense(s2) ≈ 2 * vψ atol = 1e-5

	# compress: bond cap respected and the represented chain recovered
	ψbig = mult(H, ψ, SVDCompression(trunc=truncdim(64)))
	c2 = compress(ψbig, DMRG2(maxiter=10, tol=1e-10, trunc=truncdim(8), verbosity=0))
	@test bonddim(c2) <= 8
	@test norm(todense(c2) - vHψ) / norm(vHψ) < 1e-6

	# hadamard: two-site updates handle the bond-16 exact product within the cap
	φ = randommps(ComplexF64, ds; D=4)
	χ = hadamard(ψ, φ, DMRG2(maxiter=15, tol=1e-10, trunc=truncdim(16), verbosity=0))
	refχ = vψ .* todense(φ)
	@test norm(todense(χ) - refχ) / norm(refχ) < 1e-5

	# linsolve: unitary system U·x = y recovered with a small guess (bonds grow)
	U = timeevompo(H, 0.15, WII())
	x_exact = randommps(ComplexF64, ds; D=4)
	yU = mult(U, x_exact)
	xsol = linsolve(U, yU, DMRG2(maxiter=20, tol=1e-10, trunc=truncdim(8), verbosity=0))
	@test abs(dot(xsol, x_exact)) / (norm(xsol) * norm(x_exact)) ≈ 1 atol = 1e-5

	# the DMRG2 sweeps keep the working chain right-canonical (final sweep direction)
	@test isrightcanonical(out[end-1]) || isrightcanonical(out[end])

	# NoTruncation on a small system: the two-site update is the exact local optimum,
	# so the per-pair loss is monotone in processing time (within and across sweeps) and
	# every problem converges to machine precision — this pins the local machinery
	# (targets, re-splits, environments) independently of truncation effects
	Random.seed!(777)
	Ls = 4
	dss = fill(2, Ls)
	ps = model_params(Ls)
	Hs = mpo_model(ps)
	Hsd = dense_model(ps)
	ψs = randommps(ComplexF64, dss; D=6)
	vψs = todense(ψs)
	nt = DMRG2(maxiter=30, tol=1e-14, trunc=NoTruncation())
	monotone(khist) = all(all(kh2 .>= kh1 .- 1e-8) for (kh1, kh2) in zip(khist, khist[2:end]))

	# mult MPS
	mc = MultCache(MPO(tompotensors(Hs)), ψs, svdguess_mult(Hs, ψs, 64))
	kh = iterative_compute!(mc, nt)
	@test monotone(kh)
	@test todense(mult(MPO(tompotensors(Hs)), ψs, nt)) ≈ Hsd * vψs atol = 1e-8 rtol = 1e-8

	# mult MPO·MPO
	mc = MultCache(MPO(tompotensors(Hs)), MPO(tompotensors(Hs)), svdguess_mult(Hs, Hs, 64))
	kh = iterative_compute!(mc, nt)
	@test monotone(kh)
	@test todense(mult(MPO(tompotensors(Hs)), MPO(tompotensors(Hs)), nt)) ≈ Hsd * Hsd atol = 1e-8 rtol = 1e-8

	# add
	oc = AddCache(svdguess_add([ψs, ψs], 64), [ψs, ψs])
	kh = iterative_compute!(oc, nt)
	@test monotone(kh)
	@test todense(add([ψs, ψs], nt)) ≈ 2 * vψs atol = 1e-8 rtol = 1e-8

	# compress
	xbig = mult(Hs, ψs, SVDCompression(trunc=truncdim(64)))
	cc = OverlapCache(svdguess_compress(xbig, 64), xbig)
	kh = iterative_compute!(cc, nt)
	@test monotone(kh)

	# hadamard
	φs = randommps(ComplexF64, dss; D=4)
	refχs = vψs .* todense(φs)
	hc = HadamardCache(ψs, φs, svdguess_hadamard(ψs, φs, 64))
	kh = iterative_compute!(hc, nt)
	@test monotone(kh)
	@test todense(hadamard(ψs, φs, nt)) ≈ refχs atol = 1e-8 rtol = 1e-8

	# linsolve
	Us = timeevompo(Hs, 0.15, WII())
	xs_exact = randommps(ComplexF64, dss; D=4)
	ys = mult(Us, xs_exact)
	lc = LinsolveCache(MPO(tompotensors(Us)), ys, randommps(ComplexF64, dss; D=8, normalize=false))
	kh = iterative_compute!(lc, nt)
	@test monotone(kh)
	@test abs(dot(todense(lc.ket), todense(xs_exact))) /
		  (norm(todense(lc.ket)) * norm(todense(xs_exact))) ≈ 1 atol = 1e-8
end
