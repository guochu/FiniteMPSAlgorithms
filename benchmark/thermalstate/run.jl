# Thermal-state benchmarks.
#
# Usage:
#   julia --project=. benchmark/thermalstate/run.jl          # run everything
#   julia --project=. benchmark/thermalstate/run.jl ed       # a single benchmark
#
# Available benchmarks:
#   ed_l4      L = 4,  β = 1.0 : TDVP (superoperator) and itebd vs exact diag.
#   scale_l20  L = 20, β = 0.05: TDVP (superoperator) vs itebd (no ED at this scale)
#   lowtemp    L = 10, β = 1.0 : iTEBD vs TDVP at the same bond dimension, both against
#                                exact diag. (spin-1/2 convention)
#   tdvp2_l10  L = 10, β = 1.0 : two-site TDVP from the bond-dimension-1 infinite-
#                                temperature guess (bonds grow dynamically, no
#                                `changebond!`) vs exact diag. and MPSKit's TDVP2
#   pdmrg_l10  L = 10, β = 10  : p-DMRG thermal state in its low-temperature regime vs
#                                exact diag. (ρ(β) = e^{-βH}/Z)
#   mpo_tdvp   L = 6/10        : density-operator TDVP (TDVPCache, left multiplication
#                                H·ρ) vs the vectorized left-superoperator route —
#                                strict per-step equivalence + efficiency
#
# Each route evolves the infinite-temperature state I/2^L with its thermal generator: the
# two-sided routes go to T = β/2, giving ρ(β) = e^{-βH/2}·I·e^{-βH/2}/2^L, while the
# left-multiplication route of `mpo_tdvp` goes to T = β, giving ρ(β) = e^{-βH}·I/2^L.

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "ed_l4.jl"))
include(joinpath(@__DIR__, "scale_l20.jl"))
include(joinpath(@__DIR__, "lowtemp_l10.jl"))
include(joinpath(@__DIR__, "tdvp2_l10.jl"))
include(joinpath(@__DIR__, "pdmrg_l10.jl"))
include(joinpath(@__DIR__, "mpo_tdvp.jl"))

const BENCHMARKS = Dict(
    "ed_l4" => bench_ed,
    "scale_l20" => bench_scale,
    "lowtemp_l10" => bench_lowtemp,
    "tdvp2_l10" => bench_tdvp2_l10,
    "pdmrg_l10" => bench_pdmrg_l10,
    "mpo_tdvp" => bench_mpo_tdvp,
)

function run(which::AbstractString)
    if which == "all"
        for (name, f) in BENCHMARKS
            say("="^70)
            say("benchmark: $name")
            say("="^70)
            f()
        end
    else
        haskey(BENCHMARKS, which) || error("unknown benchmark $which; available: ",
                                           join(sort(collect(keys(BENCHMARKS))), ", "))
        BENCHMARKS[which]()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run(get(ARGS, 1, "all"))
end
