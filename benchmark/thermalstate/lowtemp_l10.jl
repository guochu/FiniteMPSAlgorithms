# L = 10, β = 1 (spin-1/2 convention, S = σ/2, J = 1): itebd vs TDVP vs exact
# diagonalization at a genuinely low temperature. β = 1 keeps the purification bond
# inside a small budget (it grows exponentially in β), while E(β=1) is already well
# below the infinite-temperature value. Both routes evolve to T = β/2, i.e.
# ρ(β) = e^{-βH/2}·I·e^{-βH/2}/2^L.
function bench_lowtemp()
    Random.seed!(123)
    L = 10
    ds = fill(2, L)
    β = 1.0
    Sx = 0.5 .* _SX
    Sy = 0.5 .* _SY
    Sz = 0.5 .* _SZ
    hf = [0.4 * sin(1.3 * i + 0.2) for i in 1:L]

    # compact Hamiltonian for the vectorized routes (the exact `+` bond-sums, so the
    # vectorized chain is truncated back to the true MPO bond)
    H = prodmpo(ComplexF64, ds, [1, 2], [Sx, Sx]) + prodmpo(ComplexF64, ds, [1, 2], [Sy, Sy]) +
        prodmpo(ComplexF64, ds, [1, 2], [Sz, Sz])
    for b in 2:L-1
        H = H + prodmpo(ComplexF64, ds, [b, b+1], [Sx, Sx]) +
            prodmpo(ComplexF64, ds, [b, b+1], [Sy, Sy]) +
            prodmpo(ComplexF64, ds, [b, b+1], [Sz, Sz])
    end
    for i in 1:L
        H = H + prodmpo(ComplexF64, ds, i, Sz) * hf[i]
    end
    Hv, _ = truncate!(vectorize(H); trunc=truncdimcutoff(8, 1e-12))
    H = devectorize(Hv)
    say("compressed Boltzmann-relevant MPO bond: ", bonddim(H))
    𝕀 = identitympo(ComplexF64, ds)
    trr(ρ) = real(expectation(𝕀, ρ))

    # --- itebd: Strang brickwall of exact local gates on the vectorized chain ---
    I2 = Matrix{ComplexF64}(I, 2, 2)
    I4 = Matrix{ComplexF64}(I, 4, 4)
    ul(A) = kron(I2, A)
    ur(A) = kron(transpose(A), I2)
    pairgen(c, A, B) = c .* (kron(ul(A), ul(B)) .+ kron(ur(A), ur(B)))
    gb = [pairgen(1.0, Sx, Sx) .+ pairgen(1.0, Sy, Sy) .+ pairgen(1.0, Sz, Sz) for _ in 1:L-1]
    for i in 1:L
        h4 = hf[i] .* (ul(Sz) .+ ur(Sz))
        i > 1 && (gb[i-1] .+= (i == L ? 1.0 : 0.5) .* kron(I4, h4))
        i < L && (gb[i] .+= (i == 1 ? 1.0 : 0.5) .* kron(h4, I4))
    end
    even = [b for b in 1:L-1 if iseven(b)]
    odd = [b for b in 1:L-1 if isodd(b)]

    t = time()
    ψi = vectorize(infinite_temperature_state(ComplexF64, ds))
    trunc = truncdimcutoff(128, 1e-10)
    gate!(pos, G, τ) = apply!(GeneralGate(pos, gate_tensor(exp(Matrix(-τ * G)))), ψi; trunc=trunc)
    nst = 20
    dτ = β / 2 / nst
    for _ in 1:nst
        for b in even; gate!((b, b + 1), gb[b], dτ / 2); end
        for b in odd; gate!((b, b + 1), gb[b], dτ / 2); end
        for b in odd; gate!((b, b + 1), gb[b], dτ / 2); end
        for b in even; gate!((b, b + 1), gb[b], dτ / 2); end
    end
    ρi = devectorize(ψi)
    tr_i = trr(ρi)
    E_i = real(expectationvalue(H, ρi))
    say("itebd: ", time() - t, " s; tr/2^L = $(tr_i / 2^L)   E = $E_i   bond = ", bonddim(ψi))

    # --- TDVP on the superoperator generator ---
    # (`noise = 0`: thermal-state preparation needs exact padding — injected noise sits
    # in the exponentially amplified low-temperature sector of the cooling flow.
    # `changebond!` leaves the gauge as produced by the resize: restore the canonical
    # form externally for the TDVP sweeps.)
    t = time()
    ψ = changebond!(vectorize(infinite_temperature_state(ComplexF64, ds)); D=32, noise=0)
    restore_gauge!(ψ)
    env = DMRGCache(superoperator(H, :left) + superoperator(H, :right), ψ)
    alg = TDVP1(stepsize=-β / 2 / 20, verbosity=0)
    for _ in 1:20
        sweep!(env, alg)
    end
    ρt = devectorize(env.ket)
    tr_t = trr(ρt)
    E_t = real(expectationvalue(H, ρt))
    say("tdvp:  ", time() - t, " s; tr/2^L = $(tr_t / 2^L)   E = $E_t   bond = ", bonddim(env.ket))

    # --- ED referee ---
    Hd = todense(H)
    ex = exp(Matrix(-β * Hermitian(Hd)))
    rho_ed = ex / tr(ex)
    E_ed = real(tr(Hd * rho_ed))
    say("ED:    tr/2^L = $(real(tr(ex)) / 2^L)   E = $E_ed")

    say("E rel diff: itebd vs ED = ", abs(E_i - E_ed) / abs(E_ed),
        "   tdvp vs ED = ", abs(E_t - E_ed) / abs(E_ed))
end
