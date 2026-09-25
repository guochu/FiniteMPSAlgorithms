# Density-operator cooling (`TDVPCache` with a CanonicalMPO state: the generator acts from
# the left, ρ ↦ exp(stepsize·H)·ρ with `stepsize = -τ`) vs the vectorized route (a
# `DMRGCache` with `superoperator(H, :left)` acting on the vectorized density operator).
# Both apply the same left multiplication, the second one on the isometric (vectorized)
# chain, so with matched bond dimensions every step must agree up to Krylov/roundoff noise.
#
# Only the imaginary-time flow is meaningful here: real-time evolution of a density
# operator is two-sided, ρ(t) = e^{-i·t·H}·ρ(0)·e^{+i·t·H}, and is driven through the
# superoperator difference instead — `DMRGCache(superoperator(H, :left) -
# superoperator(H, :right), vectorize(ρ))` with `stepsize = -im·τ` (see the TDVPCache
# docstring). The same two-sided vectorized route with
# `(superoperator(H, :left) + superoperator(H, :right)) / 2` is the symmetric cooling.
function bench_mpo_tdvp()
    say("="^64)
    say("part 1: strict per-step equivalence (L = 6, imaginary time)")
    say("="^64)
    Random.seed!(123)
    L = 6
    ds = fill(2, L)
    Sx = 0.5 .* _SX
    Sy = 0.5 .* _SY
    Sz = 0.5 .* _SZ
    hf = [0.4 * sin(1.3 * i + 0.2) for i in 1:L]

    terms = OpSum(ds)
    for i in 1:L-1
        push!(terms, OpTerm(1.0, i => Sx, i + 1 => Sx))
        push!(terms, OpTerm(1.0, i => Sy, i + 1 => Sy))
        push!(terms, OpTerm(1.0, i => Sz, i + 1 => Sz))
    end
    for i in 1:L
        push!(terms, OpTerm(hf[i], i => Sz))
    end
    HH = MPOHamiltonian(terms)
    D = 16
    τ = 0.05
    nst = 5
    stepsize = -τ

    # pad BOTH initial states to the same bond profile: otherwise the two routes
    # evolve on different manifolds (bond 2^k vs the padded D) and legitimately
    # produce different TDVP trajectories. `noise = 0` keeps the per-step
    # equivalence at the roundoff level.
    ρ0 = changebond!(randommpo(ComplexF64, ds; D=D); D=D, noise=0)
    restore_gauge!(ρ0)   # both routes start from the same canonical gauge
    envA = TDVPCache(HH, ρ0)
    ψ0 = changebond!(vectorize(ρ0); D=D)
    envB = DMRGCache(superoperator(HH, :left), ψ0)
    alg = TDVP1(stepsize=stepsize, verbosity=0)
    worst = 0.0
    for step in 1:nst
        sweep!(envA, alg)
        sweep!(envB, alg)
        ρA = todense(envA.state)
        ρB = todense(devectorize(envB.ket))
        rel = norm(ρA - ρB) / norm(ρA)
        worst = max(worst, rel)
        say("  cooling step $step: ||ρ_mpo - ρ_vec||/||ρ|| = ", rel)
    end
    say("  cooling: worst per-step deviation over $nst steps = ", worst)

    # one-step accuracy of each route against the exact cooling e^{-τ·H}·ρ₀
    Hs6 = todense(MPO(tompotensors(HH)))
    ρ0d = todense(ρ0)
    exact = exp(-τ * Hs6) * ρ0d
    envA1 = TDVPCache(HH, ρ0)
    ρ0v = changebond!(vectorize(ρ0); D=D)
    envB1 = DMRGCache(superoperator(HH, :left), ρ0v)
    alg1 = TDVP1(stepsize=stepsize, verbosity=0)
    sweep!(envA1, alg1)
    sweep!(envB1, alg1)
    eA = norm(todense(envA1.state) - exact) / norm(exact)
    eB = norm(todense(devectorize(envB1.ket)) - exact) / norm(exact)
    say("  one-step error vs exact: mpo = $eA   vec = $eB")

    say("="^64)
    say("part 2: efficiency (L = 10, β = 1.0, spin-1/2, 10 steps of τ = 0.05)")
    say("="^64)
    L = 10
    ds = fill(2, L)
    Sx = 0.5 .* _SX
    Sy = 0.5 .* _SY
    Sz = 0.5 .* _SZ
    hf = [0.4 * sin(1.3 * i + 0.2) for i in 1:L]

    terms = OpSum(ds)
    for i in 1:L-1
        push!(terms, OpTerm(1.0, i => Sx, i + 1 => Sx))
        push!(terms, OpTerm(1.0, i => Sy, i + 1 => Sy))
        push!(terms, OpTerm(1.0, i => Sz, i + 1 => Sz))
    end
    for i in 1:L
        push!(terms, OpTerm(hf[i], i => Sz))
    end
    HH = MPOHamiltonian(terms)
    Hs = todense(MPO(tompotensors(HH)))
    D = 32
    nst = 10
    τ = 0.05

    # --- route A: density-operator cooling (CanonicalMPO state, left multiplication) ---
    ρ0 = changebond!(tompo(Matrix{ComplexF64}(I, 2^L, 2^L), ds); D=D, noise=0)
    restore_gauge!(ρ0)   # the sweeps need a canonical gauge
    envA = TDVPCache(HH, ρ0)
    algA = TDVP1(stepsize=-τ, verbosity=0)
    t = time()
    for _ in 1:nst
        sweep!(envA, algA)
    end
    t_mpo = time() - t
    ρA = todense(envA.state)
    E_mpo = real(tr(Hs * ρA) / tr(ρA))
    say("mpo-manifold: $nst steps in $t_mpo s ($(t_mpo / nst) s/step); bond = ",
        bonddims(envA.state), "; E = $E_mpo")

    # --- route B: vectorized cooling (CanonicalMPS state, left superoperator) ---
    ψ = changebond!(vectorize(infinite_temperature_state(ComplexF64, ds)); D=D, noise=0)
    restore_gauge!(ψ)   # the sweeps need a canonical gauge
    envB = DMRGCache(superoperator(HH, :left), ψ)
    algB = TDVP1(stepsize=-τ, verbosity=0)
    t = time()
    for _ in 1:nst
        sweep!(envB, algB)
    end
    t_vec = time() - t
    ρB = todense(devectorize(envB.ket))
    E_vec = real(tr(Hs * ρB) / tr(ρB))
    say("vectorized:   $nst steps in $t_vec s ($(t_vec / nst) s/step); bond = ",
        bonddim(envB.ket), "; E = $E_vec")

    # --- ED referee (both routes reach β_eff = τ·nst) ---
    ex = exp(Matrix(-(τ * nst) * Hermitian(Hs)))
    ρed = ex ./ tr(ex)
    E_ed = real(tr(Hs * ρed))
    say("ED (β_eff = $(τ * nst)): E = $E_ed")
    say("E rel diff: mpo vs ED = ", abs(E_mpo - E_ed) / abs(E_ed),
        "   vec vs ED = ", abs(E_vec - E_ed) / abs(E_ed),
        "   mpo vs vec = ", abs(E_mpo - E_vec) / abs(E_vec))
end
