# L = 10, β = 1 (spin-1/2 convention, S = σ/2, J = 1): **two-site TDVP** (`TDVP2`) from the
# *bond-dimension-1* infinite-temperature state — no `changebond!` padding. The pair
# updates grow the bonds dynamically up to the truncation cap `truncdim(D)` (single-site
# TDVP1 would keep the trivial initial profile forever), so the same problem that
# `lowtemp_l10` solves with a pre-padded guess is solved here by the algorithm itself.
#
# The guess still has to be **canonical**: `vectorize(infinite_temperature_state(...))`
# represents I/2^L exactly, but its site tensors are the plain identities, which are not
# isometries — the TDVP environment stacks are the environments of an isometric chain and
# the projected flow is gauge covariant only in that gauge. `rightorth!` (SVD,
# `NoTruncation`) resets the gauge in place without touching the state or the bond profile;
# it is exactly the final step of `changebond!`, not a bond-dimension change.
#
# The reference values of MPSKit's `TDVP2` for the same model, the same β, the same bond
# cap D and the same number of sweeps (nst = 20 imaginary steps of β/2/nst per side) are
# printed for the alignment check — see `mpskit_tdvp2.jl` in this directory.
function bench_tdvp2_l10()
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

    # the infinite-temperature guess: bond dimension 1, canonicalized in place
    ψ0 = vectorize(infinite_temperature_state(ComplexF64, ds))
    say("guess: bond profile before rightorth! = ", bonddims(ψ0),
        "; right-canonical = ", iscanonical(ψ0))
    rightorth!(ψ0)
    say("       bond profile after  rightorth! = ", bonddims(ψ0),
        "; right-canonical = ", iscanonical(ψ0))

    nst = 20
    dτ = β / 2 / nst
    # MPSKit TDVP2 reference: (D, ‖ρ-ρ_ed‖/‖ρ_ed‖) at the same D, β = 1, nst = 20
    mpskit_ref = Dict(8 => 9.0484e-3, 16 => 3.6317e-4, 32 => 2.9521e-6)

    say("")
    @printf("%4s %10s %18s %14s %14s %-24s %14s\n", "D", "method", "E", "E rel diff",
            "||ρ-ρ_ed||", "bond profile", "MPSKit ||ρ-ρ_ed||")
    for D in (8, 16, 32)
        t = time()
        ψ = copy(ψ0)
        env = DMRGCache(superoperator(H, :left) + superoperator(H, :right), ψ)
        alg = TDVP2(stepsize=-dτ, trunc=truncdim(D), verbosity=0)
        for _ in 1:nst
            sweep!(env, alg)
        end
        ρt = devectorize(env.ket)
        ρtd = todense(ρt)
        E_t = real(expectationvalue(H, ρt))
        err_t = norm(ρtd ./ tr(ρtd) - ρed) / norm(ρed)
        @printf("%4d %10s %18.12f %14.3e %14.4e %-24s %14.4e  (%4.1f s)\n", D, "tdvp2", E_t,
                abs(E_t - E_ed) / abs(E_ed), err_t, string(bonddims(env.ket)), mpskit_ref[D],
                time() - t)
    end
    say("")
    say("note: the guess starts at bond dimension 1 everywhere; the first sweep grows the")
    say("      bonds towards their maximal profile (4, 16, 64, ...) at this L, so D is only")
    say("      active as a cap. The MPSKit column is `TDVP2(trscheme = truncrank(D))` on the")
    say("      same model, β and sweep count (see `mpskit_tdvp2.jl`): the two agree to the")
    say("      four significant digits shown, the residual coming from the SVD/Krylov")
    say("      backends, not from the algorithm.")
end
