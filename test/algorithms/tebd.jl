# matrix in the Kronecker convention (i1 i2)',(i1 i2), i1 slowest -> documented gate
# tensor (i1', i2', i1, i2)
function gate_tensor(M::AbstractMatrix)
	d2r, d2c = size(M)
	dr, dc = isqrt(d2r), isqrt(d2c)
	(dr^2 == d2r && dc^2 == d2c) || throw(ArgumentError("dimensions must be perfect squares"))
	return permutedims(reshape(M, dr, dc, dr, dc), (2, 1, 4, 3))
end

# dense L-site matrix of the documented (i1',...,iN',i1,...,iN) gate tensor on sites (i, i+1)
function two_site_dense(op4, i, L, d::Int=2)
	M = reshape(permutedims(op4, (2, 1, 4, 3)), d^2, d^2)   # (i1 i2)',(i1 i2), i1 slowest
	I2 = Matrix{ComplexF64}(I, d, d)
	K = Matrix{ComplexF64}(I, 1, 1)
	k = 1
	while k <= L
		if k == i
			K = kron(K, M)
			k += 2
		else
			K = kron(K, I2)
			k += 1
		end
	end
	return K
end

# dense L-site matrix of the documented gate tensor on the NON-adjacent pair (i, j)
# (identities on the legs in between; supports j = i + 3, the tested case)
function two_site_dense_pair(op4, i, j, L, d::Int=2)
	j - i == 3 || throw(ArgumentError("unsupported gap"))
	δ = Matrix{ComplexF64}(I, d, d)
	# block over sites i..j; the row/column index must be ordered FASTEST-first for the
	# column-major reshape (dims run p_j, m2, m1, p_i | q_j, n2, n1, q_i), so that site i
	# is the slowest index inside the block, matching the adjacent kron embedding
	@tensor B[pj, m2, m1, p1, qj, n2, n1, q1] := op4[p1, pj, q1, qj] * δ[m1, n1] * δ[m2, n2]
	K = reshape(B, d^4, d^4)
	I2 = Matrix{ComplexF64}(I, d, d)
	return kron(vcat(fill(I2, i - 1), [K], fill(I2, L - j))...)
end


@testset "tebd" begin
	Random.seed!(55)
	# UnitaryGate rejects non-unitary input
	h = randn(4, 4) + im * randn(4, 4)
	@test_throws ArgumentError UnitaryGate((1, 2), gate_tensor(h))
	# the unitarity-check tolerance is a keyword: a loose atol admits a perturbed unitary
	Up = exp(Matrix(-im * (h + h'))) + 1e-6 * randn(4, 4) + 1e-6im * randn(4, 4)
	@test_throws ArgumentError UnitaryGate((1, 2), gate_tensor(Up))            # rejected at default tol
	UnitaryGate((1, 2), gate_tensor(Up); atol=1e-3)               # accepted with atol=1e-3
	hh = h + h'                          # hermitian → exp(-im·hh) is unitary
	U = exp(Matrix(-im * hh))
	g = UnitaryGate((1, 2), gate_tensor(U))          # ok
	@test positions(g) == (1, 2)
	# adjoint of a unitary gate is unitary and equals conj-transposed
	gt = adjoint(g)
	@test gt.op ≈ permutedims(conj(g.op), (3, 4, 1, 2))

	# GeneralGate stores the given op verbatim (no unitarity check, caller builds exp);
	# gate_tensor converts the Kronecker-convention matrix to the documented tensor
	gg = GeneralGate((2, 3), gate_tensor(exp(Matrix(-im * hh * 0.1))))
	@test gg.op ≈ permutedims(reshape(exp(Matrix(-im * hh * 0.1)), 2, 2, 2, 2), (2, 1, 4, 3)) atol = 1e-12

	# apply! of an exact unitary vs dense contraction
	L = 6
	ψ = randommps(ComplexF64, fill(2, L); D=8)
	canonicalize!(ψ)
	@test iscanonical(ψ)
	ψref = todense(ψ)
	Vd = two_site_dense(gg.op, 2, L)
	apply!(gg, ψ; trunc=NoTruncation())
	ψnew = todense(ψ)
	@test ψnew ≈ Vd * ψref atol = 1e-10
	# right-canonical form preserved without a trailing canonicalize!
	@test isrightcanonical(ψ[1])
	@test iscanonical(ψ; atol=1e-8)
	@test norm(ψ) ≈ 1 atol = 1e-10

	# a whole TEBD layer of gates keeps the state canonical (Hastings auto-canonical behavior)
	ψ2 = randommps(ComplexF64, fill(2, L); D=8)
	canonicalize!(ψ2)
	vref = todense(ψ2)
	gateAt = Dict{Int,Matrix{ComplexF64}}()
	for i in 1:2:L-1
		ggl = UnitaryGate((i, i + 1), gate_tensor(exp(Matrix(-im * hh * 0.07))))
		gateAt[i] = reshape(permutedims(ggl.op, (2, 1, 4, 3)), 4, 4)
		apply!(ggl, ψ2; trunc=NoTruncation())
	end
	@test iscanonical(ψ2; atol=1e-8)
	vlayer = copy(vref)
	for i in 1:2:L-1
		K = Matrix{ComplexF64}(I, 1, 1)
		k = 1
		while k <= L
			if k == i
				K = kron(K, gateAt[i]); k += 2
			else
				K = kron(K, Matrix{ComplexF64}(I, 2, 2)); k += 1
			end
		end
		vlayer = K * vlayer
	end
	@test todense(ψ2) ≈ vlayer atol = 1e-10

	# swap! exchanges the physical content of the neighboring sites i, i+1 (the
	# Hastings SWAP gate): the dense amplitude (x1, x2, ...) moves to (x2, x1, ...)
	ψa = randommps(ComplexF64, fill(2, L); D=8)
	canonicalize!(ψa)
	ψa0 = todense(ψa)
	n0 = norm(ψa)
	swap!(ψa, 2; trunc=NoTruncation())
	# todense flattens with site 1 the SLOWEST index, so a column-major reshape has
	# dim k = site L+1-k; exchanging sites 2,3 means exchanging dims L-1,L-2
	vt = reshape(ψa0, fill(2, L)...)
	perm = collect(1:L)
	perm[L-1], perm[L-2] = perm[L-2], perm[L-1]
	vexp = vec(permutedims(vt, Tuple(perm)))
	@test norm(todense(ψa) - vexp) / norm(vexp) < 1e-10
	@test norm(ψa) ≈ n0 atol = 1e-10
	@test isrightcanonical(ψa[1])
	@test iscanonical(ψa; atol=1e-8)

	# sequences of adjacent swaps (permutation2swaps) realize the site permutation:
	# the inverse sequence brings back the original physical content exactly
	ψb = randommps(ComplexF64, fill(2, L); D=8)
	canonicalize!(ψb)
	ψb0 = copy(ψb)
	vb0 = todense(ψb)
	perm = randperm(L)
	for b in FiniteMPSAlgorithms.permutation2swaps(perm)
		swap!(ψb, b; trunc=NoTruncation())
	end
	@test iscanonical(ψb; atol=1e-8)
	@test todense(ψb) != vb0 || isperm(perm) && perm == collect(1:L)   # content moved
	for b in FiniteMPSAlgorithms.permutation2swaps(invperm(perm))
		swap!(ψb, b; trunc=NoTruncation())
	end
	@test iscanonical(ψb; atol=1e-8)
	@test norm(todense(ψb) - vb0) / norm(vb0) < 1e-10
	@test norm(ψb) ≈ norm(ψb0) atol = 1e-10
end

@testset "general gate" begin
	Random.seed!(77)
	L = 6
	# GeneralGate accepts non-unitary input that UnitaryGate rejects
	h = randn(4, 4) + im * randn(4, 4)
	@test_throws ArgumentError UnitaryGate((1, 2), gate_tensor(h))
	gg = GeneralGate((1, 2), gate_tensor(h))
	@test gg.op ≈ permutedims(reshape(Matrix{ComplexF64}(h), 2, 2, 2, 2), (2, 1, 4, 3))
	# adjoint swaps the ket/bra index blocks and conjugates
	@test adjoint(gg).op ≈ permutedims(conj(gg.op), (3, 4, 1, 2))

	# apply! of a non-unitary gate: physical state matches the dense application,
	# and the state is re-canonicalized (Hastings alone cannot keep it canonical)
	ψ = randommps(ComplexF64, fill(2, L); D=8)
	canonicalize!(ψ)
	ψref = todense(ψ)
	apply!(gg, ψ; trunc=NoTruncation())
	K = two_site_dense(gg.op, 1, L)
	@test todense(ψ) ≈ K * ψref atol = 1e-10
	@test norm(ψ) ≈ norm(K * ψref) atol = 1e-10
	@test iscanonical(ψ; atol=1e-8)

	# a unitary GeneralGate behaves identically to UnitaryGate (plus canonicalize!)
	ψu = randommps(ComplexF64, fill(2, L); D=8)
	canonicalize!(ψu)
	ψuref = todense(ψu)
	hh = h + h'
	gu = GeneralGate((3, 4), gate_tensor(exp(Matrix(-im * hh))))
	Vd = two_site_dense(gu.op, 3, L)
	apply!(gu, ψu; trunc=NoTruncation())
	@test todense(ψu) ≈ Vd * ψuref atol = 1e-10
        @test iscanonical(ψu; atol=1e-8)
        @test norm(ψu) ≈ 1 atol = 1e-10
end

@testset "gates coverage" begin
    Random.seed!(88)
    hh = randn(4, 4) + im * randn(4, 4)
    hh = hh + hh'
    U = exp(Matrix(-im * hh))
    g = UnitaryGate((1, 2), gate_tensor(U))
    @test g isa AbstractGate
    @test GeneralGate((1, 2), gate_tensor(exp(Matrix(-im * hh * 0.1)))) isa AbstractGate
    # shift: move gate by 2 sites
    g2 = shift(g, 2)
    @test positions(g2) == (3, 4)
    @test g2.op ≈ g.op
    # shift back
    g3 = shift(g2, -2)
    @test positions(g3) == (1, 2)
    # shift a GeneralGate (possibly non-unitary)
    gg = GeneralGate((1, 2), gate_tensor(exp(Matrix(-im * hh * 0.1))))
    gg2 = shift(gg, 1)
    @test positions(gg2) == (2, 3)
end

@testset "TEBD vs exact diagonalization" begin
    Random.seed!(40)
    L = 4
    ds = fill(2, L)
    dim = 2^L
    # random hermitian nearest-neighbor couplings
    pairs = [(1, 2), (2, 3), (3, 4)]
    hs = [(randn(4, 4) + im * randn(4, 4)) for _ in pairs]
    hs = [(h + h') / 2 for h in hs]
    I2 = Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)

    # dense H = Σ_i h_(i,i+1); the kron ordering keeps site 1 the slowest index,
    # matching todense. A nearest-neighbor h is ONE kron slot spanning sites (i, i+1).
    Hd = zeros(ComplexF64, dim, dim)
    for (k, (i, j)) in enumerate(pairs)
        factors = ntuple(a -> a == i ? hs[k] : I2, L - 1)
        Hd .+= reshape(kron(factors...), dim, dim)
    end
    ψ0 = randommps(ComplexF64, ds; D=8)
    v0 = todense(ψ0)

    # a single brickwall layer on disjoint sites (1,2) and (3,4): the two gates act on
    # independent factors and commute, so one Trotter layer is EXACT
    dt = 0.37
    ψ = copy(ψ0)
    apply!(GeneralGate((1, 2), gate_tensor(exp(-im * dt * hs[1]))), ψ; trunc=truncdimcutoff(64, 1e-12))
    apply!(GeneralGate((3, 4), gate_tensor(exp(-im * dt * hs[3]))), ψ; trunc=truncdimcutoff(64, 1e-12))
    Ulayer = kron(exp(-im * dt * hs[1]), exp(-im * dt * hs[3]))
    @test norm(todense(ψ) - Ulayer * v0) / norm(v0) < 1e-12

    # second-order Trotter time evolution vs exp(-i t H) from exact diagonalization
    dt = 0.01
    nsteps = 20
    G(k, τ) = GeneralGate((pairs[k][1], pairs[k][2]), gate_tensor(exp(-im * τ * hs[k])))
    ψt = copy(ψ0)
    for _ in 1:nsteps
        for k in 1:3
            apply!(G(k, dt / 2), ψt; trunc=truncdimcutoff(64, 1e-12))
        end
        for k in 3:-1:1
            apply!(G(k, dt / 2), ψt; trunc=truncdimcutoff(64, 1e-12))
        end
    end
    ψexact = exp(-im * nsteps * dt * Hd) * v0
    @test norm(todense(ψt) - ψexact) / norm(ψexact) < 1e-3
end

@testset "long-range gates and swaps" begin
    Random.seed!(77)
    L = 6
    ψ = randommps(ComplexF64, fill(2, L); D=4)
    canonicalize!(ψ)
    v0 = todense(ψ)

    # exchange the contents of two (generally non-adjacent) sites as a sequence of
    # adjacent swaps (equivalently: permutation2swaps of the transposition (i j))
    function exchange!(x, i, j; trunc=NoTruncation())
        lo, hi = minmax(i, j)
        for b in hi-1:-1:lo
            swap!(x, b; trunc)
        end
        for b in lo+1:hi-1
            swap!(x, b; trunc)
        end
        return x
    end

    # exchange!(ψ, i, j) moves the site contents: compare against the full-space
    # permutation that swaps the corresponding bits of every basis index
    i, j = 2, 5
    P = zeros(ComplexF64, 2^L, 2^L)
    for idx in 0:2^L-1
        bits = [(idx >> (L - t)) & 1 for t in 1:L]
        bits[i], bits[j] = bits[j], bits[i]
        jdx = sum(bits[t] << (L - t) for t in 1:L)
        P[jdx+1, idx+1] = 1.0
    end
    ψs = copy(ψ)
    exchange!(ψs, i, j)
    @test iscanonical(ψs; atol=1e-8)
    @test norm(todense(ψs) - P * v0) / norm(v0) < 1e-10
    # swapping back restores the original state exactly
    exchange!(ψs, i, j)
    @test norm(todense(ψs) - v0) / norm(v0) < 1e-10

    # adjacent-pair content swap
    ψadj = copy(ψ)
    exchange!(ψadj, 3, 4)
    Padj = zeros(ComplexF64, 2^L, 2^L)
    for idx in 0:2^L-1
        bits = [(idx >> (L - t)) & 1 for t in 1:L]
        bits[3], bits[4] = bits[4], bits[3]
        jdx = sum(bits[t] << (L - t) for t in 1:L)
        Padj[jdx+1, idx+1] = 1.0
    end
    @test norm(todense(ψadj) - Padj * v0) / norm(v0) < 1e-10

    # swapping a site with itself is a no-op
    ψid = copy(ψ)
    exchange!(ψid, 3, 3)
    @test todense(ψid) ≈ v0 atol = 1e-12

    # sites with different physical dimensions: the dimension list follows the content
    dsn = [2, 3, 2, 2]
    ψn = randommps(ComplexF64, dsn; D=3)
    canonicalize!(ψn)
    vn = todense(ψn)
    exchange!(ψn, 2, 4)
    @test phydims(ψn) == [2, 2, 2, 3]
    # column-major reshape: dim k = site L+1-k (fastest first); swapping sites 2,4
    # exchanges dims 3 and 1
    vt = reshape(vn, reverse(dsn)...)
    vref = vec(permutedims(vt, (3, 2, 1, 4)))
    @test norm(todense(ψn) - vref) / norm(vref) < 1e-10

    # long-range unitary gate on a non-adjacent pair vs the dense embedding
    hh = randn(4, 4) + im * randn(4, 4)
    U = exp(Matrix(-im * (hh + hh')))
    gu = UnitaryGate((2, 5), gate_tensor(U))
    ψu = copy(ψ)
    apply!(gu, ψu; trunc=NoTruncation())
    Vu = two_site_dense_pair(gu.op, 2, 5, L)
    @test norm(todense(ψu) - Vu * v0) / norm(v0) < 1e-10
    # the content swaps are exact unitary re-gaugings: canonical form and norm preserved
    @test iscanonical(ψu; atol=1e-8)
    @test norm(ψu) ≈ 1 atol = 1e-10

    # long-range non-unitary general gate vs the dense embedding (re-canonicalized)
    gg = GeneralGate((2, 5), gate_tensor(randn(ComplexF64, 4, 4)))
    ψg = copy(ψ)
    apply!(gg, ψg; trunc=NoTruncation())
    Vg = two_site_dense_pair(gg.op, 2, 5, L)
    @test norm(todense(ψg) - Vg * v0) / norm(Vg * v0) < 1e-10
    @test iscanonical(ψg; atol=1e-8)

    # truncation keeps the bond dimension bounded
    ψt = changebond!(copy(ψ); D=8)
    apply!(gu, ψt; trunc=truncdim(4))
    @test bonddim(ψt) <= 4
end

@testset "TEBD Lindblad (vectorized density matrix) vs ED" begin
    Random.seed!(43)
    L = 4
    d4 = fill(4, L)   # vectorized chain: one site carries (po, pi), po the slower index

    # model: H = -Σ h_i X_i - J Σ Z_i Z_{i+1}; hermitian jumps √γx·X on site 1, √γz·Z on site L
    h = [0.7, -0.4, 0.9, 0.2]; J = 0.6; γx = 0.35; γz = 0.55
    X = Float64[0 1; 1 0]; Z = Float64[1 0; 0 -1]
    I2 = Matrix{ComplexF64}(I, 2, 2); I4 = Matrix{ComplexF64}(I, 4, 4)

    # --- local superoperator generators in the MPS (interleaved) index basis ---
    # per-site index s = 2(po-1) + (pi-1): po is the slower quantum number; site 1 slowest
    field_gen(i) = -h[i] .* (kron(X, I2) - kron(I2, transpose(X)))   # -i[h_i X_i, ·]

    function zz_gen(i)                                               # -i[J Z_i Z_{i+1}, ·]
        tups = collect(Iterators.product(0:1, 0:1, 0:1, 0:1))        # (po', pi', po', pi')
        sidx(t) = (2 * t[1] + t[2]) * 4 + (2 * t[3] + t[4])
        G = zeros(ComplexF64, 16, 16)
        for rp in 1:16, cp in 1:16
            a, b, a2, b2 = tups[rp]
            c, d, c2, d2 = tups[cp]
            G[sidx((a, b, a2, b2)) + 1, sidx((c, d, c2, d2)) + 1] =
                -J * (Z[a+1, c+1] * Z[a2+1, c2+1] * (b == d) * (b2 == d2)
                      - Z[b+1, d+1] * Z[b2+1, d2+1] * (a == c) * (a2 == c2))
        end
        return G
    end

    # embed a two-site 16×16 block at bond (i, i+1) into the 4^L-dimensional chain
    function embed2(G, i)
        factors = Matrix{ComplexF64}[I4 for _ in 1:i-1]
        push!(factors, G)
        append!(factors, Matrix{ComplexF64}[I4 for _ in 1:L-i-1])
        return kron(factors...)
    end

    # hermitian single-site jump √γ·L: 𝒟(ρ) = γ(LρL - ½{L²,ρ}) = γ(kron(L,L) - I4) locally
    jump_gen(Lm, γ) = γ .* (kron(Lm, Lm) .- I4)

    # ---- ED reference: full Lindblad superoperator acting on the MPS-ordered vector ----
    super = zeros(ComplexF64, 4^L, 4^L)
    for i in 1:L
        g = field_gen(i)
        super .-= im .* kron([k == i ? g : I4 for k in 1:L]...)
    end
    for i in 1:L-1
        super .-= im .* embed2(zz_gen(i), i)
    end
    for (Lm, γ, s) in [(X, γx, 1), (Z, γz, L)]
        g = jump_gen(Lm, γ)
        super .+= kron([k == s ? g : I4 for k in 1:L]...)
    end

    # ---- MPS side: exact local exponential gates, Strang splitting ----
    # single-site generators embedded into bond gates (site L via the last bond);
    # gates on different sites commute, so the dissipator factor is exact
    fraw = [1 => kron(field_gen(1), I4), 2 => kron(field_gen(2), I4),
            3 => kron(field_gen(3), I4), 3 => kron(I4, field_gen(4))]
    even = [1 => zz_gen(1), 3 => zz_gen(3)]   # bonds (1,2), (3,4)
    odd = [2 => zz_gen(2)]                    # bond (2,3)
    diss = [1 => kron(jump_gen(X, γx), I4), 3 => kron(I4, jump_gen(Z, γz))]
    tr = truncdimcutoff(64, 1e-13)

    # U(τ) = e^{-i τ ad_H} via the symmetric local-gate sequence
    #   fields(τ/2), even(τ/2), odd(τ/2), odd(τ/2), even(τ/2), fields(τ/2)
    function apply_unitary!(ψ, τ)
        ev(pos, M) = apply!(GeneralGate((pos, pos + 1), gate_tensor(exp(Matrix(-im * (τ / 2) * M)))), ψ; trunc=tr)
        for (pos, M) in fraw; ev(pos, M); end
        for (pos, M) in even; ev(pos, M); end
        for (pos, M) in odd; ev(pos, M); end
        for (pos, M) in odd; ev(pos, M); end
        for (pos, M) in even; ev(pos, M); end
        for (pos, M) in fraw; ev(pos, M); end
    end

    # Strang step: U(dt/2) · e^{𝒟 dt} · U(dt/2)
    function lindblad_step!(ψ, dt)
        apply_unitary!(ψ, dt / 2)
        for (pos, M) in diss
            apply!(GeneralGate((pos, pos + 1), gate_tensor(exp(Matrix(dt * M)))), ψ; trunc=tr)
        end
        apply_unitary!(ψ, dt / 2)
    end

    ψ = randommps(ComplexF64, d4; D=8)
    v0 = todense(ψ)
    dt, nsteps = 0.02, 12
    for _ in 1:nsteps
        lindblad_step!(ψ, dt)
    end

    v_exact = exp(super * (dt * nsteps)) * v0
    v_mps = todense(ψ)
    @test norm(v_mps - v_exact) / norm(v_exact) < 5e-3
end
