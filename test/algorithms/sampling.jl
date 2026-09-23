using FiniteMPSAlgorithms
using Test, Random, LinearAlgebra

@testset "direct sampling" begin
	Random.seed!(314)
	L = 4
	ds = fill(2, L)
	ψ = randommps(ComplexF64, ds; D=6)   # normalize=true: the modulus does not affect sampling

	# the exact Born distribution of the represented state (todense order: site 1 slowest)
	flat(𝐱) = 1 + sum((𝐱[i] - 1) * prod(ds[i+1:end]) for i in 1:L)
	p_exact = abs2.(todense(ψ))
	p_exact ./= sum(p_exact)

	# amplitude convention: 1-based indices, scaling^L included
	for _ in 1:5
		𝐱 = [rand(1:ds[i]) for i in 1:L]
		@test amplitude(ψ, 𝐱) ≈ todense(ψ)[flat(𝐱)] atol = 1e-10
		ψs = ψ * 2.0   # lmul! folds the factor into `scaling`: the represented state is 2·ψ
		@test amplitude(ψs, 𝐱) ≈ todense(ψs)[flat(𝐱)] atol = 1e-10
		@test amplitude(ψs, 𝐱) ≈ 2.0 * amplitude(ψ, 𝐱) rtol = 1e-10
	end

	tv(p, q) = 0.5 * sum(abs.(p .- q))

	# increasing the sample size converges to the exact distribution
	sizes = [200, 2_000, 20_000]
	tvs = Float64[]
	for N in sizes
		s = sample(ψ, N)
		@test length(s) == N
		@test all(all(1 <= x[i] <= ds[i] for i in 1:L) for x in s)
		p_emp = zeros(length(p_exact))
		for 𝐱 in s
			p_emp[flat(𝐱)] += 1.0 / N
		end
		push!(tvs, tv(p_emp, p_exact))
	end
	@test tvs[2] < tvs[1]
	@test tvs[3] < tvs[2]
	@test tvs[end] < 0.1

	# uninitialized Schmidt values: canonicalized in place before sampling
	ψu = randommps(ComplexF64, ds; D=4)
	unset_svectors!(ψu)
	s = sample(ψu, 2_000)
	p_exact = abs2.(todense(ψu))
	p_exact ./= sum(p_exact)
	p_emp = zeros(length(p_exact))
	for 𝐱 in s
		p_emp[flat(𝐱)] += 1.0 / 2_000
	end
	@test tv(p_emp, p_exact) < 0.1
	@test iscanonical(ψu)
end
