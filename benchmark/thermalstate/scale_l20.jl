# L = 20, β = 0.05: TDVP (superoperator) vs itebd, no ED at this scale.
# β = 0.05 keeps the vectorized thermal state's bond dimension within reach of a small
# benchmark: the exact state is the vectorization of e^{-βH}·(I/2^L)·e^{-βH}, whose
# bond grows roughly like the square of the Boltzmann-MPO bond (here ≈ 19), so larger
# β at L = 20 would need bonds ≳ 10³. Both routes evolve to T = β/2, i.e.
# ρ(β) = e^{-βH/2}·I·e^{-βH/2}/2^L.
function bench_scale()
    L = 20
    ds = fill(2, L)
    β = 0.05
    hf = [0.4 * sin(1.3 * i + 0.2) for i in 1:L]

    # Heisenberg chain with an inhomogeneous field, assembled from product terms
    # (traceless, so the infinite-temperature energy is E(0) = tr(H)/2^L = 0). The
    # exact `+` is a block-diagonal sum whose bond grows with the term count, so the
    # vectorized chain is truncated back to the true MPO bond before the
    # superoperators are built.
    function heisenberg_mpo(L, ds, hf)
        H = prodmpo(ComplexF64, ds, [1, 2], [_SX, _SX]) +
            prodmpo(ComplexF64, ds, [1, 2], [_SY, _SY]) +
            prodmpo(ComplexF64, ds, [1, 2], [_SZ, _SZ])
        for b in 2:L-1
            H = H + prodmpo(ComplexF64, ds, [b, b+1], [_SX, _SX]) +
                prodmpo(ComplexF64, ds, [b, b+1], [_SY, _SY]) +
                prodmpo(ComplexF64, ds, [b, b+1], [_SZ, _SZ])
        end
        for i in 1:L
            H = H + prodmpo(ComplexF64, ds, i, _SZ) * hf[i]
        end
        return H
    end
    H = heisenberg_mpo(L, ds, hf)
    Hv, _ = truncate!(vectorize(H); trunc=truncdimcutoff(8, 1e-12))
    H = devectorize(Hv)
    say("compressed Boltzmann-relevant MPO bond: ", bonddim(H))

    𝕀 = identitympo(ComplexF64, ds)
    trr(ρ) = real(expectation(𝕀, ρ))
    energy(ρ) = real(expectationvalue(H, ρ))

    # --- itebd: Strang brickwall of exact local gates ---
    I2 = Matrix{ComplexF64}(I, 2, 2)
    I4 = Matrix{ComplexF64}(I, 4, 4)
    ul(A) = kron(I2, A)               # A on the fast po leg
    ur(A) = kron(transpose(A), I2)    # Aᵀ on the slow pi leg
    pairgen(c, A, B) = c .* (kron(ul(A), ul(B)) .+ kron(ur(A), ur(B)))
    gb = [pairgen(1.0, _SX, _SX) .+ pairgen(1.0, _SY, _SY) .+ pairgen(1.0, _SZ, _SZ)
          for _ in 1:L-1]
    for i in 1:L
        h4 = hf[i] .* (ul(_SZ) .+ ur(_SZ))
        i > 1 && (gb[i-1] .+= (i == L ? 1.0 : 0.5) .* kron(I4, h4))
        i < L && (gb[i] .+= (i == 1 ? 1.0 : 0.5) .* kron(h4, I4))
    end
    t = time()
    ψi = vectorize(infinite_temperature_state(ComplexF64, ds))
    trunc = truncdimcutoff(64, 1e-9)
    gate!(pos, G, τ) = apply!(GeneralGate(pos, gate_tensor(exp(Matrix(-τ * G)))), ψi; trunc=trunc)
    nst = 12
    dτ = β / 2 / nst
    even = [b for b in 1:L-1 if iseven(b)]
    odd = [b for b in 1:L-1 if isodd(b)]
    Ei = Float64[]
    for _ in 1:nst
        for b in even; gate!((b, b + 1), gb[b], dτ / 2); end
        for b in odd; gate!((b, b + 1), gb[b], dτ / 2); end
        for b in odd; gate!((b, b + 1), gb[b], dτ / 2); end
        for b in even; gate!((b, b + 1), gb[b], dτ / 2); end
        push!(Ei, energy(devectorize(ψi)))
    end
    say("itebd: $nst steps in ", time() - t, " s; bond = ", bonddim(ψi), "; E trajectory = ", Ei)

    # --- TDVP on the superoperator generator: `changebond!` pads the bond profile to
    #     D = 16 headroom. `noise = 0` (plain padding) is used here on purpose: the
    #     padding directions of a β = 0.05 state sit in the exponentially amplified
    #     low-temperature sector of the cooling flow, and noise-level junk there grows
    #     into overflow within a few steps (see the mpo_tdvp equivalence benchmark for
    #     the small-padding regime where the default noise is harmless).
    t = time()
    ψ = changebond!(vectorize(infinite_temperature_state(ComplexF64, ds)); D=16, noise=0)
    env = DMRGCache(superoperator(H, :left) + superoperator(H, :right), ψ)
    alg = TDVP1(stepsize=-β / 2 / 10, verbosity=0)
    Et = Float64[]
    for _ in 1:10
        sweep!(env, alg)
        push!(Et, energy(devectorize(env.ket)))
    end
    say("tdvp: 10 steps in ", time() - t, " s; bond = ", bonddim(env.ket), "; E trajectory = ", Et)

    # both routes agree on trace, energy and local observables (no ED reference at L=20)
    ρi = devectorize(ψi)
    ρt = devectorize(env.ket)
    tr_i = trr(ρi)
    tr_t = trr(ρt)
    say("trace: itebd = $tr_i   tdvp = $tr_t   rel = ", abs(tr_i - tr_t) / tr_t)
    say("E:     itebd = ", Ei[end], "   tdvp = ", Et[end],
        "   rel = ", abs(Ei[end] - Et[end]) / abs(Et[end]))
    for (s, O, lbl) in [(3, _SZ, "<Sz3>"), (7, _SX, "<Sx7>")]
        oi = real(expectationvalue(term(s => O), ρi))
        ot = real(expectationvalue(term(s => O), ρt))
        say("$lbl: itebd = $oi   tdvp = $ot")
    end
end
