using Test
using LinearAlgebra
using Random
using TensorOperations
using FiniteMPSAlgorithms
# disambiguate from LinearAlgebra's SVD/QR/LQ factorization objects
using FiniteMPSAlgorithms: SVD, QR, QRpos, LQ, LQpos, SDD, Polar, DefaultTruncation, l_LL, r_RR

include("helpers.jl")

@testset verbose = true "FiniteMPSAlgorithms" begin
	@testset "tensorops" begin
		include("tensorops/linalg.jl")
		include("tensorops/truncation.jl")
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
