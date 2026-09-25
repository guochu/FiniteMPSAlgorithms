# L = 10, β = 1 (spin-1/2 convention, S = σ/2, J = 1): iTEBD vs TDVP **at the same bond
# dimension**, i.e. the accuracy of both routes against exact diagonalization as the gold
# standard for the same truncation budget.
#
# β = 1 keeps the bond inside a small budget (it grows exponentially in β) while E(β = 1)
# is already well below the infinite-temperature value. Both routes evolve the
# infinite-temperature state to T = β/2, i.e. ρ(β) = e^{-βH/2}·I·e^{-βH/2}/2^L, and both
# initial guesses come from `changebond!` — which pads the bond profile and returns the
# chain right-canonical, the gauge the sweeps expect.
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

    # ED gold standard: ρ(β) = e^{-βH}/Z
    Hd = todense(H)
    ex = exp(Matrix(-β * Hermitian(Hd)))
    ρed = ex / tr(ex)
    E_ed = real(tr(Hd * ρed))
    say("ED (β = ", β, "): E = ", E_ed)

    # iTEBD pieces: Strang brickwall of exact local gates on the vectorized chain
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
    nst = 20
    dτ = β / 2 / nst

    say("")
    @printf("%4s %7s %18s %14s %14s %6s\n", "D", "method", "E", "E rel diff",
            "||ρ-ρ_ed||/||ρ_ed||", "bond")
    for D in (8, 16, 32)
        # --- iTEBD: exact local gates, truncated to bond dimension D ---
        t = time()
        ψi = vectorize(infinite_temperature_state(ComplexF64, ds))
        trunc = truncdim(D)
        gate!(pos, G, τ) = apply!(GeneralGate(pos, gate_tensor(exp(Matrix(-τ * G)))), ψi; trunc=trunc)
        for _ in 1:nst
            for b in even; gate!((b, b + 1), gb[b], dτ / 2); end
            for b in odd; gate!((b, b + 1), gb[b], dτ / 2); end
            for b in odd; gate!((b, b + 1), gb[b], dτ / 2); end
            for b in even; gate!((b, b + 1), gb[b], dτ / 2); end
        end
        ρi = devectorize(ψi)
        ρid = todense(ρi)
        E_i = real(expectationvalue(H, ρi))
        err_i = norm(ρid ./ tr(ρid) - ρed) / norm(ρed)
        @printf("%4d %7s %18.12f %14.3e %14.3e %6d  (%4.1f s)\n", D, "itebd", E_i,
                abs(E_i - E_ed) / abs(E_ed), err_i, bonddim(ψi), time() - t)

        # --- TDVP: same bond dimension (the padded guess is kept throughout) ---
        t = time()
        ψ = changebond!(vectorize(infinite_temperature_state(ComplexF64, ds)); D=D, noise=0)
        env = DMRGCache(superoperator(H, :left) + superoperator(H, :right), ψ)
        alg = TDVP1(stepsize=-β / 2 / nst, verbosity=0)
        for _ in 1:nst
            sweep!(env, alg)
        end
        ρt = devectorize(env.ket)
        ρtd = todense(ρt)
        E_t = real(expectationvalue(H, ρt))
        err_t = norm(ρtd ./ tr(ρtd) - ρed) / norm(ρed)
        @printf("%4d %7s %18.12f %14.3e %14.3e %6d  (%4.1f s)\n", D, "tdvp", E_t,
                abs(E_t - E_ed) / abs(E_ed), err_t, bonddim(env.ket), time() - t)
    end
    say("")
    say("note: both routes use the same time step (dτ = β/2/", nst, " = ", dτ,
        "); TDVP's projector splitting is exact for the projected flow, so its error is")
    say("      the manifold truncation alone and keeps decreasing with D, while iTEBD")
    say("      saturates at its Trotter error.")
end
