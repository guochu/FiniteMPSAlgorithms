using FiniteMPSAlgorithms
using Test, LinearAlgebra, Random

include(joinpath(@__DIR__, "..", "helpers.jl"))

@testset "operators and expectation" begin
	Random.seed!(42)
	L = 6
	p = model_params(L)
	H = mpo_model(p)
	Hd = dense_model(p)
	# todense helper for operators
	@test maximum(abs.(todense(H) - Hd)) < 1e-12
	# identity trace
	@test tr(identitympo(Float64, fill(2, L))) ≈ 2^L
	# exact MPO·MPS vs dense
	ψ0 = prodmps(Float64, fill(2, L), fill(1, L))   # |000...⟩
	ψ0d = zeros(2^L); ψ0d[1] = 1.0
	@test todense(H * ψ0) ≈ Hd * ψ0d atol = 1e-10
	# expectation vs dense with a random canonical state
	ψ = randommps(Float64, fill(2, L); D=16)
	vψ = todense(ψ)
	@test real(dot(vψ, Hd * vψ)) ≈ real(expectation(H, ψ)) atol = 1e-7 * norm(vψ)^2
	# MPO addition vs dense
	@test real(dot(vψ, (2 * Hd) * vψ)) ≈ real(expectation(H + H, ψ)) atol = 1e-5 * norm(vψ)^2
	# local operator expectation value
	σx = Float64[0 1; 1 0]
	I2 = Matrix{Float64}(I, 2, 2)
	Ax = reshape(kron(ntuple(k -> k == 2 ? σx : I2, L)...), 2^L, 2^L)
	@test real(expectationvalue(term(2 => σx), ψ)) ≈ real(dot(vψ, Ax * vψ)) / dot(vψ, vψ) atol = 1e-8
	# OpTerm expectation: mixed-canonical shortcut vs the generic MPO path
	σz = Float64[1 0; 0 -1]
	op_single = term(2 => σx)                       # interior single-site term
	op_gap = term(-0.5, 1 => σx, 3 => σx, 6 => σz)  # gapped multi-site term touching both boundaries
	for op in (op_single, op_gap)
		mpo_op = MPO(MPOHamiltonian(OpSum(fill(2, L), [op])))
		@test real(expectation(op, ψ)) ≈ real(expectation(mpo_op, ψ)) atol = 1e-9
		@test real(expectationvalue(op, ψ)) ≈ real(expectationvalue(mpo_op, ψ)) rtol = 1e-10
	end
	# dense reference for the gapped term (site 1 slowest in the kron order, matching todense)
	Dg = -0.5 * kron(ntuple(k -> (k == 1 || k == 3) ? σx : (k == 6 ? σz : I2), L)...)
	@test real(dot(vψ, Dg * vψ)) ≈ real(expectation(op_gap, ψ)) atol = 1e-8 * norm(vψ)^2
end

@testset "operator arithmetic" begin
	Random.seed!(45)
	L = 4
	ds = fill(2, L)
	p = model_params(L)
	H = mpo_model(p)
	Hd = dense_model(p)
	SX = Float64[0 1; 1 0]
	SZ = Float64[1 0; 0 -1]
	I2c = Matrix{ComplexF64}(I, 2, 2)
	# product MPOs: exact product and exact block-diagonal sum vs dense kron
	P1 = prodmpo(ComplexF64, ds, 1, SX)
	P2 = prodmpo(ComplexF64, ds, 2, SX)
	dense1 = reshape(kron(SX, I2c, I2c, I2c), 2^L, 2^L)
	dense2 = reshape(kron(I2c, SX, I2c, I2c), 2^L, 2^L)
	@test todense(P1 * P2) ≈ dense1 * dense2 atol = 1e-12
	@test todense(P1 + P2) ≈ dense1 + dense2 atol = 1e-12
	# identity MPO and scalar multiples
	@test todense(identitympo(ComplexF64, ds) * P1) ≈ dense1 atol = 1e-12
	@test todense(P1 * (-1.5)) ≈ -1.5 * dense1 atol = 1e-12
	# MPO·MPO Hamiltonian product and sum vs dense
	@test maximum(abs.(todense(H * H) - Hd * Hd)) < 1e-10
	@test maximum(abs.(todense(H + H) - 2Hd)) < 1e-10
	# MPOHamiltonian from OpTerms equals the dense model
	@test maximum(abs.(todense(H) - Hd)) < 1e-12
	# tompotensors: dense expansion keeps the operator
	@test maximum(abs.(todense(MPO(tompotensors(H))) - Hd)) < 1e-12
end

@testset "operator fidelity" begin
	Random.seed!(46)
	L = 4
	ds = fill(2, L)
	hA = randommpo(ComplexF64, ds; D=4)
	hB = randommpo(ComplexF64, ds; D=4)
	dA, dB = todense(hA), todense(hB)
	# generic pair vs the direct dense Hilbert-Schmidt formula
	@test fidelity(hA, hB) ≈ abs(sum(conj.(dA) .* dB)) / (norm(dA) * norm(dB)) atol = 1e-10
	@test fidelity(hA, hA) ≈ 1 atol = 1e-12
	@test infidelity(hA, hA) ≈ 0 atol = 1e-12
	# global phase and external scaling are invisible to fidelity (unlike distance)
	hAs = copy(hA)
	setscaling!(hAs, 2.5)
	hAph = hA * cis(0.9)
	@test fidelity(hAs, hB) ≈ fidelity(hA, hB) atol = 1e-10
	@test fidelity(hAph, hB) ≈ fidelity(hA, hB) atol = 1e-10
	@test distance(hAs, hB) > distance(hA, hB)
	@test distance(hAph, hB) > 1e-3
end

@testset "todense/tompo roundtrip" begin
	Random.seed!(47)
	L = 4
	ds = fill(2, L)
	M = normalize(randn(ComplexF64, 2^L, 2^L))
	for order in (:msb, :lsb)
		ρ = tompo(M, ds; order)
		# exact inverse of todense in both site orders, right-canonical with spectra
		@test todense(ρ; order) ≈ M atol = 1e-12
		@test iscanonical(ρ)
	end
	# the bond spectrum matches the operator-Schmidt decomposition of the dense matrix
	ρ = tompo(M, ds)                                               # order = :big
	Mt = permutedims(reshape(M, (reverse(ds)..., reverse(ds)...)),
		(ntuple(i -> L + 1 - i, L)..., ntuple(i -> 2L + 1 - i, L)...))
	T = permutedims(Mt, ntuple(i -> isodd(i) ? (i + 1) ÷ 2 : L + i ÷ 2, 2L))
	@test schmidt_values(ρ; bond=2) ≈ svdvals(reshape(T, prod(ds[1:2])^2, :)) atol = 1e-10
	# a bond-cap scheme truncates: bonds stay within the cap, oversized caps stay exact
	@test todense(tompo(M, ds; trunc=truncdim(D=16))) ≈ M atol = 1e-10
	ρt = tompo(M, ds; trunc=truncdim(D=4))
	@test all(bonddims(ρt) .<= 4)
	@test norm(todense(ρt)) <= norm(M) * (1 + 1e-12)
end

@testset "CanonicalMPO arithmetic with scaling" begin
	Random.seed!(48)
	ρ = tompo(randn(ComplexF64, 16, 16), fill(2, 4); trunc=truncdim(D=4))
	σ = tompo(randn(ComplexF64, 16, 16), fill(2, 4); trunc=truncdim(D=4))
	setscaling!(ρ, 1.7)
	setscaling!(σ, 0.6)
	Mρ, Mσ = todense(ρ), todense(σ)
	# represented arithmetic: bilinear product, linear sum/difference and scalars
	@test todense(ρ * σ) ≈ Mρ * Mσ atol = 1e-8
	@test ρ * σ isa CanonicalMPO
	@test todense(ρ + σ) ≈ Mρ + Mσ atol = 1e-8
	@test todense(ρ - σ) ≈ Mρ - Mσ atol = 1e-8
	@test todense(2.5 * ρ) ≈ 2.5 * Mρ atol = 1e-8
	@test todense(ρ / 0.5) ≈ Mρ / 0.5 atol = 1e-8
	# mixing with a plain MPO folds the CanonicalMPO scale into the data
	h = MPO(copy(ρ.data))
	@test todense(ρ + h) ≈ Mρ + todense(h) atol = 1e-8
	@test todense(h * ρ) ≈ todense(h) * Mρ atol = 1e-8
	# operator application carries the scaling of both factors
	ψ = randommps(ComplexF64, fill(2, 4); D=4)
	@test todense(ρ * ψ) ≈ Mρ * todense(ψ) atol = 1e-8
end

@testset "MPO permute!" begin
	Random.seed!(49)
	ρ = tompo(randn(ComplexF64, 16, 16), fill(2, 4); trunc=truncdim(D=4))
	perm = [3, 1, 4, 2]
	ρp = permute(ρ, perm)
	dsT = (2, 2, 2, 2)
	Mt = reshape(todense(ρ), (dsT..., dsT...))
	ref = reshape(permutedims(Mt, (perm..., (perm .+ 4)...)), 16, 16)
	@test todense(ρp) ≈ ref atol = 1e-10
	@test iscanonical(ρp)
	ρc = copy(ρ)
	permute!(ρc, perm)
	@test todense(ρc) ≈ todense(ρp) atol = 1e-12
end

@testset "OpSum lattice" begin
	# `ds` is stored as a plain Vector{Int}
	s = OpSum([2, 2, 2])
	@test s.ds == [2, 2, 2]
	s2 = OpSum([2, 2], term(1 => _SX))
	@test length(s2) == 1
	push!(s2, term(0.5, 2 => _SZ))
	@test length(s2) == 2
	@test_throws ArgumentError push!(s2, term(0.5, 3 => _SX))
	@test_throws DimensionMismatch push!(s2, term(0.5, 1 => randn(3, 3)))
end
