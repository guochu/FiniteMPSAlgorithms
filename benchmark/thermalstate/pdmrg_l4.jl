# L = 4, β = 1 (spin-1/2 convention): p-DMRG (positive DMRG, PRB 105, 195152) thermal
# state vs exact diagonalization. p-DMRG directly minimizes F = tr(H ρ) - T S(ρ) on the
# positive ansatz ρ = Σ_τ V M_τ M_τ† V†; the center rank R must cover e^{S(β)} (≈ 11 at
# L = 4 β = 1), which makes this the feasible regime of the method — at larger L × β the
# rank grows exponentially and purification-style routes (ed_l4 / lowtemp_l10) take over.
function bench_pdmrg()
    Random.seed!(123)
    L = 4
    ds = fill(2, L)
    β = 1.0
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
    Hd = todense(MPO(tompotensors(HH)))

    t = time()
    rho = randompmpa(ComplexF64, ds; D=16, R=128)
    alg = PDMRG(maxiter=60, tol=1e-9, trunc=truncdim(16), R=128, verbosity=0)
    with_logger(NullLogger()) do
        thermalstate!(rho, HH, β, alg)
    end
    say("p-DMRG: ", time() - t, " s; tr(PMPA) = ", real(tr(rho)))

    # dense reconstruction of the represented density operator
    c = rho.center[]
    M = rho.mcenter[]
    ρp = zeros(ComplexF64, 2^L, 2^L)
    for τ in 1:size(M, 4)
        ψτ = CanonicalMPS(vcat(rho.data[1:c-1], [M[:, :, :, τ]], rho.data[c+1:end]))
        v = todense(ψτ)
        ρp .+= v * v'
    end
    trp = real(tr(ρp))

    # ED reference
    ex = exp(Matrix(-β * Hermitian(Hd)))
    ρed = ex / tr(ex)
    E_ed = real(tr(Hd * ρed))
    F_ed = E_ed - S_entropy(ρed) / β

    E_pd = real(tr(Hd * ρp) / trp)
    F_pd = E_pd - S_entropy(ρp ./ trp) / β
    say("E:  pdmrg = $E_pd   ED = $E_ed")
    say("F:  pdmrg = $F_pd   ED = $F_ed")
    say("matrix rel diff (pdmrg vs ED): ", norm(ρp - ρed) / norm(ρed))
end
S_entropy(ρ) = begin
    ev = eigvals(Hermitian(ρ))
    ev = ev[ev .> 1e-14]
    -sum(p -> p * log(p), ev)
end
