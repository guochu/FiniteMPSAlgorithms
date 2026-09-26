@testset "non-uniform per-site dimensions" begin
	# MPS/MPO state containers and the dense-chain constructors already accept differing
	# per-site dimensions; the sparse MPOHamiltonian path (OpSum/OpTerm/MPOHamiltonian) now
	# does as well, with every site tensor carrying its own `d`.

	# ---- a mixed spin-1/2 / spin-1 Heisenberg-like chain (hermitian, physical) ----
	ds = [2, 3, 2, 3]
	L = length(ds)
	sz(d) = ComplexF64[diagm(0 => [d - 1 - 2 * (k - 1) for k in 1:d] .// 2)...]
	spin_ops(d) = begin
		s = (d - 1) / 2
		Sz = zeros(ComplexF64, d, d)
		for k in 1:d
			Sz[k, k] = s - (k - 1)
		end
		Sp = zeros(ComplexF64, d, d)
		for k in 1:d-1
			Sp[k, k + 1] = sqrt(s * (s + 1) - (s - k) * (s - k + 1))
		end
		(Sz, Sp, Sp')
	end
	ops = [spin_ops(d) for d in ds]
	Jz, Jxy, hfield = 1.0, 0.6, 0.35
	terms = OpSum(ds)
	for i in 1:L
		push!(terms, OpTerm(-hfield, i => ops[i][1]))
	end
	for i in 1:L-1
		push!(terms, OpTerm(Jz, i => ops[i][1], i + 1 => ops[i + 1][1]))
		push!(terms, OpTerm(Jxy / 2, i => ops[i][2], i + 1 => ops[i + 1][3]))
		push!(terms, OpTerm(Jxy / 2, i => ops[i][3], i + 1 => ops[i + 1][2]))
	end
	# a long-range term: its gap sites carry their own dimensions
	push!(terms, OpTerm(0.2, 1 => ops[1][1], L => ops[L][1]))
	h = MPOHamiltonian(terms)
	# the OpTerm check only asks for square operators; OpSum validates against `ds`
	@test OpTerm(1.0, 1 => randn(ComplexF64, 2, 2), 2 => randn(ComplexF64, 3, 3)) isa OpTerm
	@test_throws DimensionMismatch push!(OpSum(ds), OpTerm(1.0, 1 => randn(ComplexF64, 3, 3)))

	op_at(op, i) = reshape(kron([k == i ? op : Matrix{ComplexF64}(I, d, d)
								 for (k, d) in enumerate(ds)]...), prod(ds), prod(ds))
	Hd = -hfield * sum(op_at(ops[i][1], i) for i in 1:L) +
		 Jz * sum(op_at(ops[i][1], i) * op_at(ops[i + 1][1], i + 1) for i in 1:L-1) +
		 Jxy / 2 * sum(op_at(ops[i][2], i) * op_at(ops[i + 1][3], i + 1) +
					   op_at(ops[i][3], i) * op_at(ops[i + 1][2], i + 1) for i in 1:L-1) +
		 0.2 * op_at(ops[1][1], 1) * op_at(ops[L][1], L)
	@test ophydims(h) == ds
	@test norm(todense(h) - Hd) / norm(Hd) < 1e-12
	@test norm(todense(MPO(h)) - Hd) / norm(Hd) < 1e-12
	@test norm(todense(MPOHamiltonian(tompotensors(h))) - Hd) / norm(Hd) < 1e-12

	# ---- states on the same lattice: amplitudes, expectations and the algorithms ----
	prof = [1, 2, 6, 3, 1]                    # the Schmidt-bound profile for ds
	ψ = CanonicalMPS([randn(ComplexF64, prof[i], ds[i], prof[i + 1]) for i in 1:L])
	canonicalize!(ψ)
	v = todense(ψ)
	@test abs(expectation(h, ψ) - dot(v, Hd * v)) / norm(v)^2 < 1e-10
	# a product state built on a non-uniform lattice
	ψp = prodmps(ComplexF64, ds, fill(1, L))
	@test phydims(ψp) == ds
	@test abs(expectation(h, ψp) - dot(todense(ψp), Hd * todense(ψp))) < 1e-10

	Eed = eigmin(Hermitian(Hd))
	E1, _ = ground_state(h, DMRG1(maxiter=100, verbosity=0))
	E2, _ = ground_state(h, DMRG2(maxiter=40, verbosity=0))
	@test isapprox(real(E1), Eed; rtol=1e-5)
	@test isapprox(real(E2), Eed; rtol=1e-6)

	# TDVP1/TDVP2 on the whole space (the Schmidt-bound profile) reproduce exp(-τH)
	τ = 0.05
	e1 = DMRGCache(h, copy(ψ))
	sweep!(e1, TDVP1(stepsize=-τ, verbosity=0))
	@test norm(todense(e1.ket) - exp(-τ * Hd) * v) / norm(v) < 1e-8
	e2 = DMRGCache(h, copy(ψ))
	sweep!(e2, TDVP2(stepsize=-τ, trunc=NoTruncation(), verbosity=0))
	@test norm(todense(e2.ket) - exp(-τ * Hd) * v) / norm(v) < 1e-8

	# the zero-padded product-state springboard works on a non-uniform lattice too
	ψc = prodmps(ComplexF64, ds, fill(1, L))
	v0 = todense(ψc)
	changebond!(ψc; D=6, noise=0)
	@test bonddims(ψc) == prof[2:end-1]
	ec = DMRGCache(h, ψc)
	sweep!(ec, TDVP2(stepsize=-τ, trunc=truncdim(6), verbosity=0))
	@test norm(todense(ec.ket) - exp(-τ * Hd) * v0) / norm(v0) < 1e-8

	# ---- the uniform API is unchanged ----
	X = randn(ComplexF64, 2, 2)
	@test MPOHamiltonian(4, OpTerm(1.0, 1 => X), OpTerm(0.5, 2 => X, 3 => X),
						 OpTerm(0.5, 3 => X, 4 => X)) isa MPOHamiltonian
	@test_throws DimensionMismatch MPOHamiltonian(4, OpTerm(1.0, 1 => X),
												  OpTerm(0.5, 2 => randn(ComplexF64, 3, 3)))
	@test_throws DimensionMismatch OpTerm(1.0, 1 => randn(ComplexF64, 2, 3))
end
