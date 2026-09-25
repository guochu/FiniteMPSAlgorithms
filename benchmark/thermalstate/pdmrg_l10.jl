# L = 10 at very low temperature (β = 10, deep in the ground-state regime): p-DMRG
# (positive DMRG, PRB 105, 195152: free-energy minimization on the purification ansatz
# ρ = Σ_τ V M_τ M_τ† V†, center rank R) vs exact diagonalization as the gold standard.
#
# This is the regime p-DMRG is made for. The center rank has to cover e^{S(β)}, and the
# entropy collapses onto the (small) ground-state entanglement as T → 0, so R stays tiny;
# at high temperature instead S → L·ln 2 and R would have to grow exponentially in L,
# which is what the purification-style routes (ed_l4, lowtemp_l10) are for.
#
# The Hamiltonian is the same Heisenberg chain (S = σ/2, J = 1) with the non-uniform
# longitudinal field used by the other thermal-state benchmarks.
function bench_pdmrg_l10()
    Random.seed!(123)
    L = 10
    ds = fill(2, L)
    β = 10.0
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

    # ED reference: ρ(β) = e^{-βH}/Z
    ex = exp(Matrix(-β * Hermitian(Hd)))
    ρed = ex / tr(ex)
    E_ed = real(tr(Hd * ρed))
    S_ed = von_neumann_entropy(ρed)
    F_ed = E_ed - S_ed / β
    say("ED: E = $E_ed   S = $S_ed   F = $F_ed   (e^S = ", exp(S_ed),
        ", rank(ρ) above 1e-12 = ", count(>(1e-12), eigvals(Hermitian(ρed))), ")")

    # p-DMRG: the rank only has to cover e^{S(β)}, which is small at this temperature
    R = 64
    D = 64
    alg = PDMRG(maxiter=200, tol=1e-10, trunc=truncdim(D), R=R, verbosity=0)
    rho = randompmpa(ComplexF64, ds; D=D, R=R)
    t = time()
    with_logger(NullLogger()) do
        thermalstate!(rho, HH, β, alg)
    end
    trp = real(tr(rho))
    ρp = pmpa_dense(rho)
    E_pd = real(tr(Hd * ρp) / tr(ρp))
    S_pd = von_neumann_entropy(ρp ./ tr(ρp))
    F_pd = E_pd - S_pd / β
    say("p-DMRG (R = $R, D = $D): ", time() - t, " s; tr/tr(ρ) = ", trp)

    say("E:  pdmrg = $E_pd   ED = $E_ed   rel diff = ", abs(E_pd - E_ed) / abs(E_ed))
    say("S:  pdmrg = $S_pd   ED = $S_ed")
    say("F:  pdmrg = $F_pd   ED = $F_ed   rel diff = ", abs(F_pd - F_ed) / abs(F_ed))
    say("matrix rel diff (pdmrg vs ED): ", norm(ρp - ρed) / norm(ρed))
end

# dense reconstruction of the density operator represented by a PMPA
function pmpa_dense(rho)
    L = length(rho)
    c = rho.center[]
    M = rho.mcenter[]
    ρp = zeros(ComplexF64, 2^L, 2^L)
    for τ in 1:size(M, 4)
        ψτ = CanonicalMPS(vcat(rho.data[1:c-1], [M[:, :, :, τ]], rho.data[c+1:end]))
        v = todense(ψτ)
        ρp .+= v * v'
    end
    return ρp
end

von_neumann_entropy(ρ) = begin
    ev = eigvals(Hermitian(ρ))
    ev = ev[ev .> 1e-14]
    -sum(p -> p * log(p), ev)
end
