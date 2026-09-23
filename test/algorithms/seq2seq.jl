using FiniteMPSAlgorithms
using Test, LinearAlgebra, Random



@testset "seq2seq" begin
	Random.seed!(902)
	L = 6
	N = 6
	dxs = fill(2, L)
	dys = fill(3, L)                       # rectangular map dx → dy
	# D = 2 matches the bond profile of the ground-truth MPO exactly (tight
	# parametrization -> fast ALS convergence); α tiny so the ridge only
	# conditions the local solves without biasing the fit
	alg = Seq2Seq(maxiter=25, tol=1e-12, verbosity=0, D=2, α=1e-8)

	# ground truth: random bond-2 MPO mapping dx → dy; targets y = Wtrue·x
	prof = [1, 2, 2, 2, 2, 2, 1]
	Wtrue = MPO([randn(ComplexF64, prof[i], dys[i], prof[i+1], dxs[i]) for i in 1:L])
	xs = [randommps(ComplexF64, dxs; D=4) for _ in 1:N]
	ys = [Wtrue * x for x in xs]

	W, traj = seq2seq(xs, ys, alg)
	@test ophydims(W) == dys
	@test iphydims(W) == dxs
	# phydim is only defined for square MPOTensors; the seq2seq MPO itself is rectangular
	Wr = MPO([randn(ComplexF64, 1, 3, 1, 2)])
	@test ophydim(Wr[1]) == 3 && iphydim(Wr[1]) == 2
	@test_throws DimensionMismatch phydim(Wr[1])
	@test_throws DimensionMismatch phydim(W[1])
	# per-sample residual: W·x ≈ y for every training pair
	for n in 1:N
		@test distance(W * xs[n], ys[n]) / norm(ys[n]) < 1e-6
	end
	# traj: per-sweep vectors of per-site losses (2L per sweep), all in processing-time
	# order (left sweep sites 1:L, right sweep sites L:-1:1); ALS on the data objective
	# is non-increasing throughout
	@test traj isa Vector{Vector{Float64}}
	@test all(==(2L), length.(traj))
	@test all(all(≤(1e-9), diff(kv[1:L])) for kv in traj)
	@test all(all(≤(1e-9), diff(kv[L+1:2L])) for kv in traj)
	# the losses actually decrease to the data-fit floor
	@test minimum(traj[end]) < 1e-8

	# default-algorithm form
	Wk, _ = seq2seq(xs, ys, Seq2Seq(maxiter=25, tol=1e-12, verbosity=0, D=2, α=1e-8))
	@test distance(Wk * xs[1], ys[1]) / norm(ys[1]) < 1e-6

	# dimension mismatch: a y with wrong physical dimensions
	ys_bad = vcat(ys[1:end-1], [randommps(ComplexF64, dxs; D=4)])
	@test_throws DimensionMismatch seq2seq(xs, ys_bad, alg)
end

@testset "seq2seq! (in-place)" begin
	Random.seed!(903)
	L = 5
	N = 16                                 # over-determined system: robust ALS fit
	dxs = fill(2, L)
	dys = fill(3, L)                       # rectangular map dx → dy
	alg = Seq2Seq(maxiter=40, tol=1e-12, verbosity=0, D=2, α=1e-8)
	prof = vcat(1, fill(2, L-1), 1)

	# ground truth: random bond-2 MPO; targets y = Wtrue·x
	Wtrue = MPO([randn(ComplexF64, prof[i], dys[i], prof[i+1], dxs[i]) for i in 1:L])
	xs = [randommps(ComplexF64, dxs; D=4) for _ in 1:N]
	ys = [Wtrue * x for x in xs]

	# fit a caller-provided MPO in place: W is the initial guess AND the result
	# (small random entries keep the ALS sweeps out of local minima)
	Wfit = MPO([0.1 .* randn(ComplexF64, prof[i], dys[i], prof[i+1], dxs[i]) for i in 1:L])
	Wbefore = [copy(A) for A in Wfit.data]
	traj = seq2seq!(Wfit, xs, ys, alg)
	@test traj isa Vector{Vector{Float64}}
	@test all(==(2L), length.(traj))
	# the site tensors were updated in place
	@test any(!isapprox(A1, A2; atol=1e-12) for (A1, A2) in zip(Wfit.data, Wbefore))
	# the fit converged to the ground truth
	for n in 1:N
		@test distance(Wfit * xs[n], ys[n]) / norm(ys[n]) < 1e-6
	end

	# scaled inputs are folded into the site tensors at entry
	xs_s = [x * 2.0 for x in xs]
	ys_s = [Wtrue * x for x in xs_s]
	Wfit2 = MPO([0.1 .* randn(ComplexF64, prof[i], dys[i], prof[i+1], dxs[i]) for i in 1:L])
	seq2seq!(Wfit2, xs_s, ys_s, alg)
	@test distance(Wfit2 * xs[1], ys[1]) / norm(ys[1]) < 1e-4
end

@testset "seq2seq adaptive (oracle)" begin
        Random.seed!(904)
        L = 4
        dxs = fill(2, L)
        dys = fill(3, L)
        prof = vcat(1, fill(2, L - 1), 1)
        Wtrue = MPO([randn(ComplexF64, prof[i], dys[i], prof[i+1], dxs[i]) for i in 1:L])
        pairfun = x -> Wtrue * x

        # adaptive data enrichment: the oracle is queried for the worst-predicted inputs
        W, info = seq2seq(pairfun, dxs, dys,
                Seq2Seq(D=2, α=1e-8, nbuffer=16, nadd=4, maxiter=15, tol=1e-10, verbosity=0))
        @test info.npairs > 16
        @test info.rounds >= 1
        @test info.maxerr < 1e-5
        # generalization on fresh random inputs
        for _ in 1:5
                x = randommps(ComplexF64, dxs; D=4, normalize=true)
                @test distance(W * x, Wtrue * x) / norm(Wtrue * x) < 1e-5
        end
end
