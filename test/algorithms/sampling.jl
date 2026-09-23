using FiniteMPSAlgorithms
using Test, Random, LinearAlgebra

@testset "direct sampling" begin
	Random.seed!(314)
	L = 4
	ds = fill(2, L)
	ψ = randommps(ComplexF64, ds; D=6)   # normalize=true: the modulus does not affect sampling

	# the exact Born distribution of the represented state
	p_exact = abs2.(todense(ψ))
	p_exact ./= sum(p_exact)

	tv(p, q) = 0.5 * sum(abs.(p .- q))

	# increasing the sample size converges to the exact distribution
	sizes = [200, 2_000, 20_000]
	tvs = Float64[]
	for N in sizes
		s = sample(ψ, N)
		@test length(s) == N
		@test all(all(1 <= x[i] <= ds[i] for i in 1:L) for x in s)
		counts = Dict{NTuple{L,Int},Float64}()
		for 𝐱 in s
			counts[𝐱] = get(counts, 𝐱, 0.0) + 1.0
		end
		# empirical distribution in todense's ordering (site 1 the slowest index)
		p_emp = zeros(length(p_exact))
		for 𝐱 in s
			ix = 1
			for i in 1:L
				ix += (𝐱[i] - 1) * prod(ds[i+1:end])
			end
			p_emp[ix] += 1.0 / N
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
		ix = 1
		for i in 1:L
			ix += (𝐱[i] - 1) * prod(ds[i+1:end])
		end
		p_emp[ix] += 1.0 / 2_000
	end
	@test tv(p_emp, p_exact) < 0.1
	@test iscanonical(ψu)
end
