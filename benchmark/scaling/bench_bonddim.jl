# Bond-dimension scaling benchmark: time / allocation / retained-memory exponents
#
# For every engine the cost should grow as D^3 (environments, local Krylov) or D^4
# (two-site SVD-type updates with an exact D^2 bond); memory must follow the same
# exponents. An exponent clearly above that signals a bad contraction order (an
# intermediate tensor carrying an avoidable extra D factor).
#
# Covered engines (no VUMPS here — this package is finite-chain):
#   mult       (h·ψ, SVD compression and variational DMRG1)
#   hadamard   (ψa ⊙ ψb, SVD compression)
#   add        (ψa + ψb, SVD compression)
#   DMRG1      (ground state, ALS + local Lanczos)
#   DMRG2      (ground state, two-site with bond growth)
#   excited    (one-sided projected DMRG1)
#
# Run:  julia --project=. bench_bonddim.jl [L] [D1 D2 ...]

using FiniteMPSAlgorithms
using LinearAlgebra
using Logging
using Printf
using Random

disable_logging(Warn)

const L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32
const Ds = length(ARGS) >= 2 ? [parse(Int, a) for a in ARGS[2:end]] : [8, 16, 32, 64]

σx = Float64[0 1; 1 0]
σy = ComplexF64[0 -im; im 0]
σz = Float64[1 0; 0 -1]
I2 = Matrix{ComplexF64}(I, 2, 2)

# hand-built bond-4 Heisenberg MPO (channels: 1 = nothing open, 2/3/4 = X/Y/Z open);
# a fixed, small MPO bond χ keeps the D-scaling clean (OpSum's Schur form would give χ ~ 3L)
function heisenberg_mpo(L)
	X = Matrix{ComplexF64}(σx); Y = Matrix{ComplexF64}(σy); Z = Matrix{ComplexF64}(σz)
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

"wall time, total allocated bytes, retained (live) bytes of `f()`"
function measure(f)
	GC.gc(); GC.gc()
	live0 = Base.gc_bytes()
	t = @elapsed alloc = @allocated f()
	GC.gc()
	live1 = Base.gc_bytes()
	return (time=t, alloc=alloc, live=max(live1 - live0, 0))
end

"log–log slope of y over the D sweep (least squares)"
function fitexp(xs, ys)
	ok = [i for i in eachindex(ys) if isfinite(ys[i]) && ys[i] > 0]
	length(ok) < 2 && return NaN
	lx = log.(xs[ok]); ly = log.(ys[ok])
	mx = sum(lx) / length(lx); my = sum(ly) / length(ly)
	return sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2)
end

h = heisenberg_mpo(L)
chid = maximum(bonddims(h))
@printf "L = %d, MPO bond ≤ %d, D sweep = %s\n\n" L chid Ds

# workload definitions: f(D) runs the engine once with bond cap D
engines = Dict{String, Function}(
	"mult-var" => D -> begin
		mult(h, randommps(ComplexF64, fill(2, L); D=D), DMRG1(maxiter=3, tol=1e-14, D=D, verbosity=0))
		nothing
	end,
	"hadamard-var" => D -> begin
		ψa = randommps(ComplexF64, fill(2, L); D=D)
		ψb = randommps(ComplexF64, fill(2, L); D=D)
		hadamard(ψa, ψb, DMRG1(maxiter=3, tol=1e-14, D=D, verbosity=0))
		nothing
	end,
	"add-var" => D -> begin
		ψa = randommps(ComplexF64, fill(2, L); D=D)
		ψb = randommps(ComplexF64, fill(2, L); D=D)
		add([ψa, ψb], DMRG1(maxiter=3, tol=1e-14, D=D, verbosity=0))
		nothing
	end,
	"linsolve" => D -> begin
		linsolve(h, randommps(ComplexF64, fill(2, L); D=D), ALSLinSolve(maxiter=3, tol=1e-14, D=D, verbosity=0))
		nothing
	end,
	"dmrg1" => D -> begin
		ψ = randommps(ComplexF64, fill(2, L); D=D)
		ground_state!(ψ, h, DMRG1(maxiter=3, tol=1e-14, D=D, verbosity=0))
		nothing
	end,
	"dmrg2" => D -> begin
		ψ = randommps(ComplexF64, fill(2, L); D=D)
		ground_state!(ψ, h, DMRG2(maxiter=2, tol=1e-14, trunc=truncdimcutoff(D, 1e-12), verbosity=0))
		nothing
	end,
	"excited" => D -> begin
		ψgs = randommps(ComplexF64, fill(2, L); D=D)
		ψ = randommps(ComplexF64, fill(2, L); D=D)
		excited_state!(ψ, h, [ψgs], DMRG1(maxiter=3, tol=1e-14, D=D, verbosity=0))
		nothing
	end,
)
const ORDER = ["mult-var", "hadamard-var", "add-var", "linsolve", "dmrg1", "dmrg2", "excited"]

# compile everything once at the smallest D
println("compiling...")
for (name, f) in engines
	f(Ds[1])
end
println("done\n")

results = Dict{String, Dict{Int, NamedTuple}}()
for name in ORDER
	results[name] = Dict{Int, NamedTuple}()
end

for D in Ds
	for name in ORDER
		r = measure(() -> engines[name](D))
		results[name][D] = r
		@printf "%-10s D = %4d : t = %9.3f s   alloc = %8.2f MiB   liveΔ = %8.2f MiB\n" name D r.time r.alloc / 2^20 r.live / 2^20
		flush(stdout)
	end
	println()
end

println("="^76)
@printf "%-10s %8s   %10s %12s %12s\n" "engine" "D-range" "t [s]" "α_time" "α_alloc"
println("-"^76)
for name in ORDER
	ts = [results[name][D].time for D in Ds]
	as = [results[name][D].alloc for D in Ds]
	αt = fitexp(Ds, ts)
	αa = fitexp(Ds, as)
	@printf "%-10s %3d-%-4d  %10.2f %12.2f %12.2f\n" name Ds[1] Ds[end] ts[end] αt αa
end
println("="^76)
println("""
expected exponents (MPO bond χ = 4 ≪ D):
  mult-var    ~ 3   (ALS: env transfers O(d χ D^3), local Lanczos on d·D space)
  hadamard-var ~ 3-4 (three-chain envs over (D, D, D) bonds; fused pair is D^4·d)
  add-var     ~ 3   (ALS sum)
  linsolve    ~ 3   (ALS normal equations, env transfers O(d χ D^3))
  dmrg1       ~ 3   (environments O(d χ D^3), local Lanczos on d·D space)
  dmrg2       ~ 3-4 (two-site Heff O(d^2 χ D^3) + SVD of (dD)^2)
  excited     ~ 3   (DMRG1 + per-site projectors)
allocations should track the same exponents; anything ≳ 4 (or 5 for hadamard) means a
bad contraction order producing avoidable D^4/D^5 intermediates.""")
