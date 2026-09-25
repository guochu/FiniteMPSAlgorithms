# MPO-manifold TDVP (MPOTDVPCache: the generator MPO multiplies the CanonicalMPO state
# directly) vs the vectorized TDVP (the superoperator generator acts on the
# vectorized CanonicalMPS purification).
#
# The two flows are the same linear map with the same stepsize when the vectorized
# generator is the *half* superoperator 𝒦½ = (S_L + S_R)/2:
#   MPOTDVPCache, imaginary stepsize -im·τ:  one sweep applies e^{-τ/2·H}·ρ·e^{-τ/2·H}
#   vectorized,  imaginary stepsize -im·τ:   one sweep applies exp(-τ·𝒦½)|ρ⟩ = same
#   real time stepsizes +τ (MPOTDVPCache commutator flow) pair with the *full*
#   superoperator difference 𝒦_diff = S_L - S_R (TDVP1's real stepsizes carry the -i).
# With matched bond dimensions both integrators discretize the identical
# Dirac-Frenkel projected flow, so every step must agree up to Krylov/roundoff noise.
function bench_mpo_tdvp()
    say("="^64)
    say("part 1: strict per-step equivalence (L = 6, imaginary + real time)")
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

    for (tag, stepsize, split) in
        [("imaginary (cooling)", -im * τ, 0.5), ("real (unitary)", τ, 1.0)]
        # pad BOTH initial states to the same bond profile: otherwise the two routes
        # evolve on different manifolds (bond 2^k vs the padded D) and legitimately
        # produce different TDVP trajectories. `noise = 0` keeps the per-step
        # equivalence at the roundoff level.
        ρ0 = changebond!(randommpo(ComplexF64, ds; D=D); D=D, noise=0)
        restore_gauge!(ρ0)   # both routes start from the same canonical gauge
        envA = MPOTDVPCache(HH, ρ0)
        𝒦 = split == 0.5 ?
            (superoperator(HH, :left) + superoperator(HH, :right)) / 2 :
            superoperator(HH, :left) - superoperator(HH, :right)
        ψ0 = changebond!(vectorize(ρ0); D=D)
        envB = DMRGCache(𝒦, ψ0)
        alg = TDVP1(stepsize=stepsize, verbosity=0)
        worst = 0.0
        for step in 1:nst
            sweep!(envA, alg)
            sweep!(envB, alg)
            ρA = todense(envA.rho)
            ρB = todense(devectorize(envB.ket))
            rel = norm(ρA - ρB) / norm(ρA)
            worst = max(worst, rel)
            say("  $tag step $step: ||ρ_mpo - ρ_vec||/||ρ|| = ", rel)
        end
        say("  $tag: worst per-step deviation over $nst steps = ", worst)

        # one-step accuracy of each route against the exact map
        # (imaginary: e^{-τ/2·H}ρe^{-τ/2·H}; real: e^{-iτH}ρe^{+iτH})
        Hs6 = todense(MPO(tompotensors(HH)))
        ρ0d = todense(ρ0)
        if split == 0.5
            exact = exp(-τ / 2 * Hermitian(Hs6)) * ρ0d * exp(-τ / 2 * Hermitian(Hs6))
        else
            exact = exp(-im * τ * Hermitian(Hs6)) * ρ0d * exp(+im * τ * Hermitian(Hs6))
        end
        envA1 = MPOTDVPCache(HH, ρ0)
        ρ0v = changebond!(vectorize(ρ0); D=D)
        envB1 = DMRGCache(𝒦, ρ0v)
        alg1 = TDVP1(stepsize=stepsize, verbosity=0)
        sweep!(envA1, alg1)
        sweep!(envB1, alg1)
        eA = norm(todense(envA1.rho) - exact) / norm(exact)
        eB = norm(todense(devectorize(envB1.ket)) - exact) / norm(exact)
        say("  one-step error vs exact: mpo = $eA   vec = $eB")
    end

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

    # --- route A: MPO-manifold TDVP (CanonicalMPO state) ---
    ρ0 = changebond!(tompo(Matrix{ComplexF64}(I, 2^L, 2^L), ds); D=D, noise=0)
    restore_gauge!(ρ0)   # the sweeps need a canonical gauge
    envA = MPOTDVPCache(HH, ρ0)
    algA = TDVP1(stepsize=-im * τ, verbosity=0)
    t = time()
    for _ in 1:nst
        sweep!(envA, algA)
    end
    t_mpo = time() - t
    ρA = todense(envA.rho)
    E_mpo = real(tr(Hs * ρA) / tr(ρA))
    say("mpo-manifold: $nst steps in $t_mpo s ($(t_mpo / nst) s/step); bond = ",
        bonddims(envA.rho), "; E = $E_mpo")

    # --- route B: vectorized TDVP (CanonicalMPS purification) ---
    ψ = changebond!(vectorize(infinite_temperature_state(ComplexF64, ds)); D=D, noise=0)
    restore_gauge!(ψ)   # the sweeps need a canonical gauge
    envB = DMRGCache((superoperator(HH, :left) + superoperator(HH, :right)) / 2, ψ)
    algB = TDVP1(stepsize=-im * τ, verbosity=0)
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
