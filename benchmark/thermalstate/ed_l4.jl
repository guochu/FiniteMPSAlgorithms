# L = 4, β = 1: TDVP (superoperator) and itebd vs exact diagonalization.
# Evolves the infinite-temperature state I/2^L with the thermal generator
# 𝒦 = kron(H, I) + kron(I, Hᵀ) to T = β/2, giving ρ(β) = e^{-βH/2}·I·e^{-βH/2}/2^L.
function bench_ed()
    Random.seed!(42)
    L = 4
    ds = fill(2, L)
    p = model_params(L)
    H = mpo_model(p)
    Hd = dense_model(p)
    β = 1.0

    𝒦 = superoperator(H, :left) + superoperator(H, :right)

    # --- TDVP route ---
    # (`noise = 0`: for thermal-state preparation the padding must be exact — noise
    # injected here sits in the exponentially amplified low-temperature sector.
    # `changebond!` leaves the gauge as produced by the resize: restore the canonical
    # form externally for the TDVP sweeps.)
    t = time()
    ψ = changebond!(vectorize(infinite_temperature_state(ComplexF64, ds)); D=16, noise=0)
    restore_gauge!(ψ)
    env = DMRGCache(𝒦, ψ)
    alg = TDVP1(stepsize=-im * β / 2 / 80, verbosity=0)
    for _ in 1:80
        sweep!(env, alg)
    end
    ρ = devectorize(env.ket)
    tr_tdvp = real(expectation(identitympo(ComplexF64, ds), ρ))
    E_tdvp = real(expectationvalue(MPO(tompotensors(H)), ρ))
    say("TDVP:  E = $E_tdvp   tr/2^L = $(tr_tdvp / 2^L)   (", time() - t, " s)")

    # --- ED reference ---
    rho_ed = exp(-β * Matrix(Hermitian(Hd)))
    Z = tr(rho_ed)
    rho_ed ./= Z
    E_ed = real(tr(Hd * rho_ed))
    S_ed = real(-tr(rho_ed * log(rho_ed)))
    say("ED:    E = $E_ed   tr/2^L = $(Z / 2^L)   F = $(E_ed - S_ed / β)")

    # local observables
    I2 = Matrix{ComplexF64}(I, 2, 2)
    Hd_mpo = MPO(tompotensors(H))
    for (s, O) in [(2, _SZ), (3, _SX)]
        Os = reshape(kron(ntuple(k -> k == s ? O : I2, L)...), 2^L, 2^L)
        o_mps = real(expectationvalue(term(s => O), ρ))
        o_ed = real(tr(Os * rho_ed))
        say("  <S$(s)>: TDVP = $o_mps   ED = $o_ed")
    end

    # --- itebd route: the same flow Trotter-split into exact local gates ---
    # per-term generators on the fused spaces (the per-site index is (po, pi) with po
    # the FAST leg, matching `vectorize`): the left action is ul(A) = kron(I₂, A), the
    # right action ur(A) = kron(Aᵀ, I₂); a two-site term c·A_iB_j contributes
    # c·(kron(ul(A), ul(B)) + kron(ur(A), ur(B))) on its pair (first pair position the
    # slowest index). The one-site field terms fold into the adjacent bond generators
    # (full weight at the edges, half in the bulk), so the gate generators sum to 𝒦
    # exactly and the Strang brickwall stays second order.
    t = time()
    I4 = Matrix{ComplexF64}(I, 4, 4)
    ul(A) = kron(I2, A)               # A on the fast po leg
    ur(A) = kron(transpose(A), I2)    # Aᵀ on the slow pi leg
    pairgen(c, A, B) = c .* (kron(ul(A), ul(B)) .+ kron(ur(A), ur(B)))
    gb = [pairgen(-p.J1[i], _SZ, _SZ) for i in 1:L-1]      # NN bonds (i, i+1)
    gnn = [pairgen(-p.J2[i], _SY, _SY) for i in 1:L-2]     # NNN pairs (i, i+2)
    for i in 1:L
        h4 = -p.hs[i] .* (ul(_SX) .+ ur(_SX))
        i > 1 && (gb[i-1] .+= (i == L ? 1.0 : 0.5) .* kron(I4, h4))
        i < L && (gb[i] .+= (i == 1 ? 1.0 : 0.5) .* kron(h4, I4))
    end

    ψi = vectorize(infinite_temperature_state(ComplexF64, ds))
    trunc = truncdimcutoff(64, 1e-12)
    gate!(pos, G, τ) = apply!(GeneralGate(pos, gate_tensor(exp(Matrix(-τ * G)))), ψi; trunc=trunc)
    nst = 20
    dτ = β / 2 / nst
    for _ in 1:nst
        gate!((1, 2), gb[1], dτ / 2); gate!((3, 4), gb[3], dτ / 2)
        gate!((2, 3), gb[2], dτ / 2)
        gate!((1, 3), gnn[1], dτ / 2); gate!((2, 4), gnn[2], dτ / 2)
        gate!((1, 3), gnn[1], dτ / 2); gate!((2, 4), gnn[2], dτ / 2)
        gate!((2, 3), gb[2], dτ / 2)
        gate!((1, 2), gb[1], dτ / 2); gate!((3, 4), gb[3], dτ / 2)
    end
    say("itebd: (", time() - t, " s)")

    # full density matrices: both routes vs ED
    ρt = todense(ρ)
    ρt ./= tr(ρt)
    ρit = todense(devectorize(ψi))
    ρit ./= tr(ρit)
    say("matrix rel diff: TDVP vs ED = ", norm(ρt - rho_ed) / norm(rho_ed),
        "   itebd vs ED = ", norm(ρit - rho_ed) / norm(rho_ed),
        "   TDVP vs itebd = ", norm(ρt - ρit) / norm(ρit))
end
