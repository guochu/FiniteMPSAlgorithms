using FiniteMPSAlgorithms
using Test, LinearAlgebra, Random

include(joinpath(@__DIR__, "..", "helpers.jl"))

@testset "CanonicalMPO entanglement" begin
	Random.seed!(73)
	L = 6
	ρ = randommpo(ComplexF64, fill(2, L); D=8)
	spec = entanglement_spectrum(ρ; bond=3)
	sv = schmidt_values(ρ; bond=3)
	@test spec ≈ abs2.(sv)
	@test sum(spec) ≈ 1 atol = 1e-10
	# von Neumann entropy of the operator-chain Schmidt spectrum
	@test entanglement_entropy(ρ; bond=3) ≈ -sum(spec .* log.(spec)) atol = 1e-8
	# Rényi α=2
	@test entanglement_entropy(ρ; bond=3, α=2) ≈ -log(sum(abs2, spec)) atol = 1e-8
end

@testset "observables (states)" begin
	Random.seed!(81)
	L = 6
	p = model_params(L)
	H = mpo_model(p)
	Hd = dense_model(p)
	ψ = randommps(ComplexF64, fill(2, L); D=8)
	vψ = todense(ψ)

	# expectation vs dense contraction
	@test real(dot(vψ, Hd * vψ)) ≈ real(expectation(H, ψ)) atol = 1e-8 * norm(vψ)^2
	# expectationvalue is normalized: divides by ⟨ψ|ψ⟩ and is scale invariant
	@test real(expectationvalue(H, ψ)) ≈ real(dot(vψ, Hd * vψ)) / dot(vψ, vψ) atol = 1e-8
	ψs = ψ * 3.7
	@test real(expectationvalue(H, ψs)) ≈ real(expectationvalue(H, ψ)) rtol = 1e-10
	# the raw expectation carries the external scale (scaling^(2L) with the full factor)
	@test real(expectation(H, ψs)) ≈ 3.7^2 * real(expectation(H, ψ)) rtol = 1e-8

	# pure-state density matrix and the infinite-temperature state
	ρ = DensityOperator(ψ)
	@test tr(ρ) ≈ 1 atol = 1e-8
	@test real(expectationvalue(MPO(H), ρ)) ≈ real(expectationvalue(H, ψ)) rtol = 1e-6
	ρ∞ = infinite_temperature_state(ComplexF64, fill(2, L))
	@test tr(ρ∞) ≈ 1 atol = 1e-10
	# unnormalized mixed state: expectation is raw tr(h·ρ), expectationvalue divides by tr(ρ)
	ρs = ρ∞ * 5.0
	@test real(expectation(MPO(H), ρs)) ≈ 5.0 * real(expectation(MPO(H), ρ∞)) rtol = 1e-10
	@test real(expectationvalue(MPO(H), ρs)) ≈ real(expectationvalue(MPO(H), ρ∞)) rtol = 1e-10

	# TraceCache: reusing the identity transfer environments across observables
	t1 = term(1.5, 2 => _SX)
	t2 = term(-0.4, 1 => _SZ, 3 => _SZ)
	t3 = term(0.7, 6 => _SY)
	cache = TraceCache(ρ∞)
	for t in (t1, t2, t3)
		@test expectation(t, ρ∞, cache) ≈ expectation(t, ρ∞) rtol = 1e-10
		@test expectationvalue(t, ρ∞, cache) ≈ expectationvalue(t, ρ∞) rtol = 1e-10
	end
	# the cache itself carries the scaled trace of ρ (the per-site scaling is folded in)
	@test scalar(cache.left[L+1]) ≈ tr(ρ∞) atol = 1e-10
end

@testset "expectationvalue scaling stability" begin
	# `expectation` materializes scaling^L powers (debug tool: may overflow), while
	# `expectationvalue` cancels the scale exactly and must stay finite and correct
	Random.seed!(91)
	L = 500
	ds = fill(2, L)
	SZ = Float64[1 0; 0 -1]
	ψ = prodmps(ComplexF64, ds, fill(1, L))   # |111...1⟩ product state
	setscaling!(ψ, 50.0)          # represented norm = 50^500 ≈ 1e850 (beyond Float64)
	@test !isfinite(expectation(identitympo(ComplexF64, ds), ψ))  # the debug tool overflows
	@test real(expectationvalue(identitympo(ComplexF64, ds), ψ)) ≈ 1 atol = 1e-10

	# a nontrivial observable on the same exploding state: ⟨1|σz|1⟩ = 1 at every site
	SZm = prodmpo(ComplexF64, ds, div(L, 2), SZ)
	@test real(expectationvalue(SZm, ψ)) ≈ 1 atol = 1e-12

	# the same stability for mixed states and OpTerms (per-site scaling folded into the
	# trace environments: large but bounded, so no intermediate overflow)
	ρ = infinite_temperature_state(ComplexF64, fill(2, 8))
	setscaling!(ρ, 1e30)
	t = term(1.5, 2 => _SX, 4 => _SZ)
	cache = TraceCache(ρ)
	v1 = real(expectationvalue(t, ρ, cache))
	setscaling!(ρ, 3.0)           # changing the scale must not change the ratio
	@test real(expectationvalue(t, ρ, TraceCache(ρ))) ≈ v1 rtol = 1e-12
	@test v1 ≈ real(expectationvalue(t, infinite_temperature_state(ComplexF64, fill(2, 8)))) rtol = 1e-12
end

@testset "todense/tomps roundtrip" begin
	Random.seed!(83)
	L = 4
	ds = fill(2, L)
	v = normalize(randn(ComplexF64, 2^L))
	for order in (:msb, :lsb)
		ψ = tomps(v, ds; order)
		# exact inverse of todense in both site orders, left-canonical with spectra
		@test todense(ψ; order) ≈ v atol = 1e-12
		@test iscanonical(ψ)
		# the recorded Schmidt spectrum matches the dense reshaping at bond 2
		vt = order === :big ?
			permutedims(reshape(v, reverse(ds)...), reverse(ntuple(i -> i, L))) :
			reshape(v, Tuple(ds))
		@test schmidt_values(ψ; bond=2) ≈ svdvals(reshape(vt, prod(ds[1:2]), :)) atol = 1e-10
		# the strict canonical gauge carries over: expectation values on the chain match
		# the dense contraction (chain site 1 is the first kron factor for :msb and the
		# last one for :lsb)
		SZ = Float64[1 0; 0 -1]
		I2 = Matrix{Float64}(I, 2, 2)
		Szsite1 = order === :msb ? kron(SZ, I2, I2, I2) : kron(I2, I2, I2, SZ)
		@test real(expectationvalue(term(1 => SZ), ψ)) ≈
			real(dot(v, Szsite1 * v)) / real(dot(v, v)) atol = 1e-10
	end
	# a bond-cap scheme truncates: bonds stay within the cap, oversized caps stay exact
	@test todense(tomps(v, ds; trunc=truncdim(D=8))) ≈ v atol = 1e-12
	ψt = tomps(v, ds; trunc=truncdim(D=2))
	@test all(bonddims(ψt) .<= 2)
	@test norm(todense(ψt)) <= norm(v) * (1 + 1e-12)
end

@testset "sum / copyphydims / permute" begin
	Random.seed!(85)
	L = 4
	ds = fill(2, L)
	ψA = randommps(ComplexF64, ds; D=4)
	# Base.sum: all amplitudes of the represented state
	@test sum(ψA) ≈ sum(todense(ψA)) rtol = 1e-10
	@test sum(ψA * 2.3) ≈ 2.3 * sum(ψA) rtol = 1e-10

	# copyphydims: the MPO·MPS mult reproduces the Hadamard product
	ψB = randommps(ComplexF64, ds; D=4)
	ρ = copyphydims(ψA)
	@test bonddims(ρ) == bonddims(ψA)
	@test ophydims(ρ) == iphydims(ρ) == phydims(ψA)
	@test scaling(ρ) == scaling(ψA)
	@test svectors_uninitialized(ρ)
	@test todense(ρ * ψB) ≈ todense(⊙(ψA, ψB)) atol = 1e-10

	# permute! / permute: site contents reordered, canonical form preserved
	perm = [3, 1, 4, 2]
	ψp = permute(ψA, perm)
	vt = reshape(todense(ψA), Tuple(ds))
	@test reshape(todense(ψp), Tuple(ds)) ≈ permutedims(vt, perm) atol = 1e-10
	@test iscanonical(ψp)
	ψc = copy(ψA)
	permute!(ψc, perm)
	@test todense(ψc) ≈ todense(ψp) atol = 1e-12
end
