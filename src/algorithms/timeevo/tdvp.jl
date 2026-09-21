# TDVP1: single-site time-dependent variational principle, reusing the sweep interface.
# One `sweep!(env, alg::TDVP1)` advances the state by `alg.stepsize` (Strang splitting:
# left half-step + right half-step). `stepsize = -im*τ` gives imaginary-time evolution.
#
# Bond-tensor updates follow MPSKit's `C_hamiltonian`: both environments entering the
# bond effective Hamiltonian are anchored at the *same* MPO bond (the bond where the
# center tensor lives). The sweeps therefore keep the environment center at the sweep
# start side and fold the neighboring site into the on-the-fly opposite environment.

# rebuild the environment stack with the center at the given position
function _retarget_center!(env::DMRGCache, center::Integer)
	if env.center[] != center
		fresh = DMRGCache(env.H, env.ket; center)
		copy!(env.hstorage, fresh.hstorage)
		env.center[] = center
	end
	return env
end

"""
	TDVP1(; stepsize, D=Defaults.D, ishermitian=true, verbosity=Defaults.verbosity)

Configuration of single-site TDVP. `stepsize` is the time step (complex allowed;
`stepsize = -im*τ` evolves in imaginary time). `ishermitian` selects the KrylovKit
exponentiate driver (Lanczos for hermitian, Arnoldi otherwise).
"""
Base.@kwdef struct TDVP1{S<:Number} <: TimeEvolutionAlgorithm
	stepsize::S
	D::Int = Defaults.D
	ishermitian::Bool = true
	verbosity::Int = Defaults.verbosity
end
TDVP1(stepsize::Number; kwargs...) = TDVP1(; stepsize, kwargs...)

"""
	leftsweep!(env::DMRGCache, alg::TDVP1)

Left half of one TDVP1 time step: every site tensor (including the last) evolves with
`exp(+dt/2·H_eff)` and every bond tensor with `exp(-dt/2·H_bond)` (complement space,
both environments anchored at the bond).
"""
function leftsweep!(env::DMRGCache, alg::TDVP1)
	L = length(env.ket)
	_retarget_center!(env, 1)
	dt = alg.stepsize
	t = -im * dt / 2
	for s in 1:L-1
		heff = Heff(env.H[s], env.hstorage[s], env.hstorage[s+1])
		x, _ = exponentiate(y -> ac_prime(y, heff), t, env.ket[s]; ishermitian=alg.ishermitian,
							tol=Defaults.tol, krylovdim=25, maxiter=100)
		env.ket[s] = x
		q, r = leftorth!(env.ket[s], (1, 2), (3,))
		env.ket[s] = q
		v = r
		updateleft!(env, s)
		# complement-space evolution on the bond: both environments anchored at bond s
		# (the right one folds in site s+1)
		hright = _updateright(env.hstorage[s+2], env.ket[s+1], env.H[s+1], env.ket[s+1])
		v, _ = exponentiate(y -> c_prime(y, env.hstorage[s+1], hright), -t, v;
							ishermitian=alg.ishermitian, tol=Defaults.tol, krylovdim=25, maxiter=100)
		env.ket[s+1] = _contract_first(env.ket[s+1], v)
	end
	heff = Heff(env.H[L], env.hstorage[L], env.hstorage[L+1])
	env.ket[L], _ = exponentiate(y -> ac_prime(y, heff), t, env.ket[L];
								 ishermitian=alg.ishermitian, tol=Defaults.tol, krylovdim=25, maxiter=100)
	return Float64[]
end

"""
	rightsweep!(env::DMRGCache, alg::TDVP1)

Right half of one TDVP1 time step (symmetric to `leftsweep!`).
"""
function rightsweep!(env::DMRGCache, alg::TDVP1)
	L = length(env.ket)
	_retarget_center!(env, L)
	dt = alg.stepsize
	t = -im * dt / 2
	for s in L:-1:2
		heff = Heff(env.H[s], env.hstorage[s], env.hstorage[s+1])
		x, _ = exponentiate(y -> ac_prime(y, heff), t, env.ket[s]; ishermitian=alg.ishermitian,
							tol=Defaults.tol, krylovdim=25, maxiter=100)
		env.ket[s] = x
		l, q = rightorth!(env.ket[s], (1,), (2, 3))
		env.ket[s] = q
		v = l
		updateright!(env, s)
		# complement-space evolution on the bond: both environments anchored at bond s-1
		# (the left one folds in site s-1)
		hleft = _updateleft(env.hstorage[s-1], env.ket[s-1], env.H[s-1], env.ket[s-1])
		v, _ = exponentiate(y -> c_prime(y, hleft, env.hstorage[s]), -t, v;
							ishermitian=alg.ishermitian, tol=Defaults.tol, krylovdim=25, maxiter=100)
		env.ket[s-1] = _contract_last(env.ket[s-1], v)
	end
	heff = Heff(env.H[1], env.hstorage[1], env.hstorage[2])
	env.ket[1], _ = exponentiate(y -> ac_prime(y, heff), t, env.ket[1];
								 ishermitian=alg.ishermitian, tol=Defaults.tol, krylovdim=25, maxiter=100)
	return Float64[]
end

"""
	sweep!(env::DMRGCache, alg::TDVP1)

One full TDVP1 time step of size `alg.stepsize` (`leftsweep!` + `rightsweep!`).
Time evolution loops are driven by the caller: repeat `sweep!(env, alg)`.
"""
sweep!(env::DMRGCache, alg::TDVP1) = begin
	leftsweep!(env, alg)
	rightsweep!(env, alg)
	return env
end
