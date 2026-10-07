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
	# a gapped multi-site term touching both boundaries; untouched sites propagate
	# their string channels with explicit identities (a term supported on interior
	# sites only leaves all-scalar boundary sites, which cannot carry a phydim)
	op_gap = term(-0.5, 1 => σx, 3 => σx, 6 => σz)
	for op in (op_gap,)
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
	@test maximum(abs.(todense(MPO(H) * MPO(H)) - Hd * Hd)) < 1e-10
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
	@test distance(hAph, hB) > 1.0e-3
	# the same interface for two MPOHamiltonians
	t1 = term(1.0, 1 => _SX) + term(0.5, 2 => _SZ) + term(0.25, 3 => _SX, 4 => _SX)
	t2 = term(1.0, 1 => _SZ) + term(0.5, 2 => _SX) + term(0.25, 3 => _SX, 4 => _SX)
	h1 = MPOHamiltonian(t1)
	h2 = MPOHamiltonian(t2)
	@test fidelity(h1, h2) isa Real
	@test infidelity(h1, h1) ≈ 0 atol = 1e-12
end

@testset "operator norm/dot stability" begin
	Random.seed!(48)
	L = 4
	ds = fill(2, L)
	hA = randommpo(ComplexF64, ds; D=4)
	# the Hilbert-Schmidt norm and dot vs the dense matrix (scaling included, per site)
	@test norm(hA) ≈ norm(todense(hA)) rtol = 1e-12
	dA = todense(hA)
	@test dot(hA, hA) ≈ tr(dA' * dA) rtol = 1e-12
	# the dot includes the per-site scaling of both operands
	setscaling!(hA, 2.5)
	@test dot(hA, hA) ≈ 2.5^2L * tr(dA' * dA) rtol = 1e-12
	setscaling!(hA, 1.0)
	# vanishing data: ⟨h|h⟩ = 0 exactly, norm returns 0 (no DomainError from sqrt)
	h0 = MPO([zeros(ComplexF64, 1, ds[i], 1, ds[i]) for i in 1:L])
	@test norm(h0) == 0.0
	# subnormal external scale: the per-site s² underflows ⟨h|h⟩ to (roundoff of) zero —
	# the real part can round to a tiny negative, the norm clamps it to 0
	ρ = randommpo(ComplexF64, ds; D=4)
	setscaling!(ρ, 1e-170)
	@test norm(ρ) == 0.0
	# moderate scaling: norm = scaling^L · norm(data at scaling 1)
	setscaling!(ρ, 0.01)
	@test norm(ρ) ≈ 0.01^L rtol = 1e-12
end

@testset "operator dot overflow stability" begin
	# fidelity is computed from the raw (scale-free) contractions: finite and
	# independent of the external scaling, even when scaling^L itself overflows
	hA = randommpo(ComplexF64, fill(2, 8); D=2)
	hB = randommpo(ComplexF64, fill(2, 8); D=2)
	setscaling!(hA, 1e150)
	setscaling!(hB, 1e150)
	@test (scaling(hA) * scaling(hB))^8 == Inf
	hA0, hB0 = copy(hA), copy(hB)
	setscaling!(hA0, 1.0)
	setscaling!(hB0, 1.0)
	@test isfinite(fidelity(hA, hB))
	@test fidelity(hA, hB) ≈ fidelity(hA0, hB0) rtol = 1e-12
	# dot/norm apply the scaling per site: with shrunken data the contractions stay
	# finite while the naive scaling^L-at-the-end evaluation overflows (deterministic
	# positive data, so no random-phase cancellation enters the magnitudes)
	L = 400
	a = CanonicalMPO([fill(0.3, 1, 2, 1, 2) for _ in 1:L]; scaling=2.5)
	b = CanonicalMPO([fill(0.3, 1, 2, 1, 2) for _ in 1:L]; scaling=2.5)
	@test (scaling(a) * scaling(b))^L == Inf
	d = dot(a, b)
	@test isfinite(d)
	n = norm(a)
	@test isfinite(n) && n > 0
	# log-scale consistency: dot/norm factor as scaling^L · raw value
	a0, b0 = copy(a), copy(b)
	setscaling!(a0, 1.0)
	setscaling!(b0, 1.0)
	@test isapprox(log(abs(d)), log(abs(dot(a0, b0))) + L * log(2.5 * 2.5); rtol=1e-6)
	@test isapprox(log(n), log(norm(a0)) + L * log(2.5); rtol=1e-6)
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
	# represented arithmetic: bilinear product, linear sum and scalars
	@test todense(ρ * σ) ≈ Mρ * Mσ atol = 1e-8
	@test ρ * σ isa CanonicalMPO
	@test todense(ρ + σ) ≈ Mρ + Mσ atol = 1e-8
	@test ρ + σ isa CanonicalMPO
	@test scaling(ρ + σ) == 1
	@test todense(2.5 * ρ) ≈ 2.5 * Mρ atol = 1e-8
	@test todense(ρ / 0.5) ≈ Mρ / 0.5 atol = 1e-8
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

@testset "operator adjoint" begin
	Random.seed!(50)
	L = 4
	ds = fill(2, L)
	# MPO: the site-reversed, per-site-adjointed chain vs the dense matrix adjoint
	hA = MPO(randommpo(ComplexF64, ds; D=4).data)
	@test todense(adjoint(hA)) ≈ adjoint(todense(hA)) atol = 1e-12
	hAA = adjoint(adjoint(hA))
	@test hAA.data == hA.data
	# CanonicalMPO: the external scaling folds into the data
	ρ = randommpo(ComplexF64, ds; D=4)
	setscaling!(ρ, 1.7)
	@test todense(adjoint(ρ)) ≈ adjoint(todense(ρ)) atol = 1e-12
	# OpTerm: conjugated coefficient, adjointed operators, involution
	σx = Float64[0 1; 1 0]
	σy = ComplexF64[0 -im; im 0]
	t = OpTerm(1.5 - 0.3im, [1, 3], [σx, σy])
	ta = adjoint(t)
	@test ta.coeff == conj(t.coeff)
	@test ta.positions == t.positions
	@test ta.operators == adjoint.(t.operators)
	@test adjoint(ta) == t
	# OpSum: term-wise adjoint; the represented operators match the dense adjoint
	s = t + OpTerm(0.5, 2 => σy)
	sa = adjoint(s)
	@test length(sa.data) == 2
	@test all(k -> sa.data[k] == adjoint(s.data[k]), 1:2)
	@test todense(MPOHamiltonian(adjoint(s))) ≈ adjoint(todense(MPOHamiltonian(s))) atol = 1e-12
end

@testset "OpSum lattice" begin
	# `ds` is stored as a plain Vector{Int} and inferred from the operators
	s = term(1 => _SX) + term(0.5, 2 => _SZ)
	@test s.ds == [2, 2]
	@test length(s.data) == 2
	# a wider term grows the lattice (new sites carry the first term's dimension)
	s3 = s + term(0.5, 3 => _SX)
	@test s3.ds == [2, 2, 2]
	# conflicting operator dimensions on one site are rejected
	@test_throws DimensionMismatch s + term(0.5, 1 => randn(3, 3))
	# incremental build from an explicit lattice: `OpSum(ds) += term`
	s4 = OpSum([2, 2])
	s4 += term(1 => _SX)
	s4 += term(-0.5, 2 => _SZ)
	@test s4.ds == [2, 2] && length(s4.data) == 2
	@test_throws DimensionMismatch s4 + term(1 => randn(3, 3))
	# a wider term grows the lattice
	s5 = s4 + term(0.5, 3 => _SX)
	@test s5.ds == [2, 2, 2]
end
