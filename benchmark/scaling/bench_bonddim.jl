# Bond-dimension scaling benchmark: time / allocation / retained-memory exponents
#
# For every engine the cost should grow as D^3 (environments, local Krylov) or D^4
# (two-site SVD-type updates with an exact D^2 bond); memory must follow the same
# exponents. An exponent clearly above that signals a bad contraction order (an
# intermediate tensor carrying an avoidable extra D factor).
#
# Covered engines (no VUMPS here - this package is finite-chain):
#   mult       (h*psi, SVD compression and variational DMRG1)
#   hadamard   (psia . psib, SVD compression)
#   add        (psia + psib, SVD compression)
#   DMRG1      (ground state, ALS + local Lanczos)
#   DMRG2      (ground state, two-site with bond growth)
#   excited    (one-sided projected DMRG1)
#
# MEMORY GUARD - this script must never OOM the machine:
#   * RSS is polled every 100 ms; the process hard-aborts (exit 42) the moment
#     it exceeds MEMCAP GiB (env var, default 16).
#   * If julia was started without --heap-size-hint, the script re-execs itself
#     with one, so the GC collects while the heap approaches the hint instead of
#     letting short-lived large temporaries balloon RSS.
#
# Run:  julia --project=. bench_bonddim.jl [L] [D1 D2 ...]      (MEMCAP=<GiB> to override)

using FiniteMPSAlgorithms
using LinearAlgebra
using Logging
using Printf
using Random

disable_logging(Warn)

const L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32
const Ds = length(ARGS) >= 2 ? [parse(Int, a) for a in ARGS[2:end]] : [8, 16, 32, 64]

# ----------------------------- memory guard -----------------------------
const MEMCAP = haskey(ENV, "MEMCAP") ? parse(Float64, ENV["MEMCAP"]) : 16.0
const PEAKRSS = Ref{Float64}(0.0)

"hard-abort the process (exit 42) once RSS exceeds MEMCAP GiB"
function check_rss()
    rss = Sys.maxrss() / 2^30                       # Sys.maxrss() is in bytes
    PEAKRSS[] = max(PEAKRSS[], rss)
    rss > MEMCAP || return
    println(stderr, "MEMCAP: RSS = ", round(rss; digits=2), " GiB > MEMCAP = ",
            MEMCAP, " GiB — hard abort")
    flush(stderr)
    ccall(:exit, Cvoid, (Cint,), 42)
    return
end

if Base.JLOptions().heap_size_hint == 0 && !isempty(PROGRAM_FILE)
    # no heap hint: re-exec with one so the GC collects before RSS balloons
    proj = something(Base.active_project(), abspath(joinpath(@__DIR__, "..", "..")))
    hint = string(ceil(Int, MEMCAP), "G")
    println("no --heap-size-hint given; re-executing with --heap-size-hint=", hint)
    flush(stdout)
    jl = joinpath(Sys.BINDIR, "julia")
    cmd = pipeline(`$jl --project=$proj --heap-size-hint=$hint $(abspath(PROGRAM_FILE)) $ARGS`;
                   stdout=stdout, stderr=stderr)
    p = run(cmd; wait=false)
    while process_running(p)
        sleep(0.05)
    end
    p.exitcode == 42 && println(stderr, "child run was aborted by the MEMCAP guard")
    exit(p.exitcode)
end

"run f() in a task while the caller polls RSS every 100 ms; the polling sleeps
double as the yield points that let the task (and the check) actually run"
function run_guarded(f)
    tsk = @task f()
    schedule(tsk)
    while !istaskdone(tsk)
        sleep(0.1)
        check_rss()
    end
    return fetch(tsk)                               # rethrows the task's error, if any
end

check_rss()
# -------------------------------------------------------------------------

sx = Float64[0 1; 1 0]
sy = ComplexF64[0 -im; im 0]
sz = Float64[1 0; 0 -1]
I2 = Matrix{ComplexF64}(I, 2, 2)

# hand-built bond-4 Heisenberg MPO (channels: 1 = nothing open, 2/3/4 = X/Y/Z open);
# a fixed, small MPO bond chi keeps the D-scaling clean (OpSum's Schur form gives chi ~ 3L)
function heisenberg_mpo(L)
    X = Matrix{ComplexF64}(sx); Y = Matrix{ComplexF64}(sy); Z = Matrix{ComplexF64}(sz)
    bulk = zeros(ComplexF64, 4, 2, 4, 2)
    for (c, O) in ((2, X), (3, Y), (4, Z))
        bulk[1, :, c, :] .= O        # open an X/Y/Z term
        bulk[c, :, 1, :] .= O        # close it on the neighbour
    end
    bulk[1, :, 1, :] .= I2           # carry the "nothing open" channel
    data = Vector{Array{ComplexF64, 4}}(undef, L)
    data[1] = reshape(bulk[1, :, :, :], 1, 2, 4, 2)   # row: only the openings
    data[L] = reshape(bulk[:, :, 1, :], 4, 2, 1, 2)   # column: only the closings
    for i in 2:L-1
        data[i] = copy(bulk)
    end
    return MPO(data)
end

"wall time, total allocated bytes, retained (live) bytes of f() (RSS-guarded)"
function measure(f)
    GC.gc(); GC.gc()
    live0 = Base.gc_live_bytes()
    alloc0 = Base.gc_bytes()
    # time f() from inside the task: the polling loop only sees 100 ms granularity
    inner = @task begin
        t0 = time_ns()
        f()
        (time_ns() - t0) / 1e9
    end
    schedule(inner)
    while !istaskdone(inner)
        sleep(0.1)
        check_rss()
    end
    t = fetch(inner)
    GC.gc()
    return (time=t, alloc=Base.gc_bytes() - alloc0, live=max(Base.gc_live_bytes() - live0, 0))
end

"log-log slope of y over the D sweep (least squares)"
function fitexp(xs, ys)
    ok = [i for i in eachindex(ys) if isfinite(ys[i]) && ys[i] > 0]
    length(ok) < 2 && return NaN
    lx = [log(xs[i]) for i in ok]
    ly = [log(ys[i]) for i in ok]
    mx = sum(lx) / length(lx)
    my = sum(ly) / length(ly)
    sxy = sum((lx[i] - mx) * (ly[i] - my) for i in eachindex(lx))
    sxx = sum((lx[i] - mx) * (lx[i] - mx) for i in eachindex(lx))
    return sxy / sxx
end

h = heisenberg_mpo(L)
chid = maximum(bonddims(h))
@printf "L = %d, MPO bond = %d, D sweep = %s, MEMCAP = %g GiB (peak so far %.2f GiB)\n\n" L chid Ds MEMCAP PEAKRSS[]

# workload definitions: f(D) runs the engine once with bond cap D
engines = [
    ("mult-var", D -> begin
		mult(h, randommps(ComplexF64, fill(2, L); D=D), DMRG1(maxiter=3, tol=1e-14, D=D, verbosity=0))
		nothing
	end),
	("mult-dmrg2", D -> begin
		mult(h, randommps(ComplexF64, fill(2, L); D=D), DMRG2(maxiter=2, tol=1e-14, trunc=truncdimcutoff(D, 1e-12), verbosity=0))
		nothing
	end),
	("hadamard-var", D -> begin
		psia = randommps(ComplexF64, fill(2, L); D=D)
		psib = randommps(ComplexF64, fill(2, L); D=D)
		hadamard(psia, psib, DMRG1(maxiter=3, tol=1e-14, D=D, verbosity=0))
		nothing
	end),
	("hadamard-dmrg2", D -> begin
		psia = randommps(ComplexF64, fill(2, L); D=D)
		psib = randommps(ComplexF64, fill(2, L); D=D)
		hadamard(psia, psib, DMRG2(maxiter=2, tol=1e-14, trunc=truncdimcutoff(D, 1e-12), verbosity=0))
		nothing
	end),
    ("add-var", D -> begin
        psia = randommps(ComplexF64, fill(2, L); D=D)
        psib = randommps(ComplexF64, fill(2, L); D=D)
        add([psia, psib], DMRG1(maxiter=3, tol=1e-14, D=D, verbosity=0))
        nothing
    end),
    ("linsolve", D -> begin
        linsolve(h, randommps(ComplexF64, fill(2, L); D=D), ALSLinSolve(maxiter=3, tol=1e-14, D=D, verbosity=0))
        nothing
    end),
    ("dmrg1", D -> begin
        psi = randommps(ComplexF64, fill(2, L); D=D)
        ground_state!(psi, h, DMRG1(maxiter=3, tol=1e-14, D=D, verbosity=0))
        nothing
    end),
    ("dmrg2", D -> begin
        psi = randommps(ComplexF64, fill(2, L); D=D)
        ground_state!(psi, h, DMRG2(maxiter=2, tol=1e-14, trunc=truncdimcutoff(D, 1e-12), verbosity=0))
        nothing
    end),
    ("excited", D -> begin
        psigs = randommps(ComplexF64, fill(2, L); D=D)
        psi = randommps(ComplexF64, fill(2, L); D=D)
        excited_state!(psi, h, [psigs], DMRG1(maxiter=3, tol=1e-14, D=D, verbosity=0))
        nothing
    end),
]

results = Dict(name => Dict{Int, NamedTuple}() for (name, _) in engines)

# compile everything once at the smallest D (also RSS-guarded)
println("compiling...")
for (name, f) in engines
    run_guarded(() -> f(Ds[1]))
end
println("done\n")

for D in Ds
    for (name, f) in engines
        peak0 = PEAKRSS[]
        r = measure(() -> f(D))
        results[name][D] = r
        @printf "%-13s D = %4d : t = %9.3f s   alloc = %8.2f MiB   live = %8.2f MiB   rss = %6.2f GiB\n" name D r.time r.alloc / 2^20 r.live / 2^20 max(PEAKRSS[] - peak0, 0.0)
        flush(stdout)
    end
    println()
end

println("="^76)
@printf "peak RSS over the whole run: %.2f GiB (MEMCAP = %g GiB)\n" PEAKRSS[] MEMCAP
println("-"^76)
@printf "%-13s %8s   %10s %12s %12s\n" "engine" "D-range" "t [s]" "a_time" "a_alloc"
println("-"^76)
for (name, _) in engines
    ts = [results[name][D].time for D in Ds]
    as = [results[name][D].alloc for D in Ds]
    at = fitexp(Ds, ts)
    aa = fitexp(Ds, as)
    @printf "%-13s %3d-%-4d  %10.2f %12.2f %12.2f\n" name Ds[1] Ds[end] ts[end] at aa
end
println("="^76)
println("""
expected exponents (MPO bond chi = 4 << D):
  mult-var      ~ 3   (ALS: env transfers O(d chi D^3), local Lanczos on d*D space)
  hadamard-var  ~ 3-4 (three-chain envs over (D, D, D) bonds; site updates O(D^3))
  add-var       ~ 3   (ALS sum)
  linsolve      ~ 3   (ALS normal equations, env transfers O(d chi D^3))
  dmrg1         ~ 3   (environments O(d chi D^3), local Lanczos on d*D space)
  dmrg2         ~ 3-4 (two-site Heff O(d^2 chi D^3) + SVD of (dD)^2)
  excited       ~ 3   (DMRG1 + per-site projectors)
allocations should track the same exponents; anything >> 4 (or 5 for hadamard) means a
bad contraction order producing avoidable D^4/D^5 intermediates.""")
