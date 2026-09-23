using Test
using LinearAlgebra
using FiniteMPSAlgorithms

@testset verbose = true "FiniteMPSAlgorithms" begin
	@testset "tensorops" begin
		include("tensorops/tensorops.jl")
	end
	@testset "states" begin
		include("states/structures.jl")
		include("states/observables.jl")
	end
	@testset "operators" begin
		include("operators/mpo.jl")
		include("operators/longrangeop.jl")
		include("operators/sparsempo.jl")
		include("operators/vectorize.jl")
	end
	@testset "algorithms" begin
		include("algorithms/arithmetics.jl")
		include("algorithms/dmrg.jl")
		include("algorithms/timeevo.jl")
		include("algorithms/tebd.jl")
		include("algorithms/seq2seq.jl")
		include("algorithms/sampling.jl")
		include("algorithms/experimental.jl")
	end
end
