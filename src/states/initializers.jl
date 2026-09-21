# initializers for CanonicalMPS / CanonicalMPO

"""
	max_bonddims(ds, D)

The maximal allowed bond-dimension profile for physical dimensions `ds` under the cap `D`.
"""
function max_bonddims(ds::AbstractVector{Int}, D::Int)
	L = length(ds)
	prof = ones(Int, L + 1)
	for i in 1:L
		prof[i+1] = min(D, prof[i] * ds[i])
	end
	return prof
end

"""
	prodmps(::Type{T}, ds, states) -> CanonicalMPS
	prodmps(::Type{T}, vectors) -> CanonicalMPS

Product states: one-hot basis states (`states[i]` is the occupied basis index of site `i`),
or general product states from per-site amplitude vectors.
"""
function prodmps(::Type{T}, ds::AbstractVector{Int}, states::AbstractVector{<:Integer}) where {T<:Number}
	(length(ds) == length(states)) || throw(DimensionMismatch())
	data = [begin
		a = zeros(T, 1, d, 1)
		a[1, states[i], 1] = 1
		a
	end for (i, d) in enumerate(ds)]
	return CanonicalMPS(data)
end

function prodmps(::Type{T}, vectors::AbstractVector{<:AbstractVector}) where {T<:Number}
	data = [reshape(convert(Vector{T}, collect(v)), 1, length(v), 1) for v in vectors]
	return CanonicalMPS(data)
end

"""
	randommps(::Type{T}, ds; D=Defaults.D, normalize=true) -> CanonicalMPS

Random MPS with bond-dimension profile `max_bonddims(ds, D)`, brought to canonical
form (right-canonical with initialized Schmidt values). With `normalize = true` the
state is normalized (total norm 1); otherwise the data keeps its natural magnitude and
the external scale stays in the `scaling` field.
"""
function randommps(::Type{T}, ds::AbstractVector{Int}; D::Int=Defaults.D, normalize::Bool=true) where {T<:Number}
	L = length(ds)
	prof = max_bonddims(ds, D)
	data = Vector{Array{T,3}}(undef, L)
	for i in 1:L
		dl = i == 1 ? 1 : prof[i]
		dr = i == L ? 1 : prof[i+1]
		data[i] = randn(T, dl, ds[i], dr)
	end
	ψ = CanonicalMPS(data)
	canonicalize!(ψ)
	normalize && lmul!(1 / norm(ψ), ψ)
	return ψ
end
randommps(ds::AbstractVector{Int}; kwargs...) = randommps(Float64, ds; kwargs...)
randommps(::Type{T}, L::Int; d::Int=2, kwargs...) where {T<:Number} = randommps(T, fill(d, L); kwargs...)

"""
	randommpo(::Type{T}, ds; D=Defaults.D, normalize=true) -> CanonicalMPO

Random canonical MPO-form state (bond-dimension profile from `max_bonddims`), brought
to canonical form. With `normalize = true` the state is normalized; otherwise the
external scale stays in the `scaling` field.
"""
function randommpo(::Type{T}, ds::AbstractVector{Int}; D::Int=Defaults.D, normalize::Bool=true) where {T<:Number}
	L = length(ds)
	prof = max_bonddims(ds, D)
	data = Vector{Array{T,4}}(undef, L)
	for i in 1:L
		dl = i == 1 ? 1 : prof[i]
		dr = i == L ? 1 : prof[i+1]
		data[i] = randn(T, dl, ds[i], dr, ds[i])
	end
	ρ = CanonicalMPO(data)
	canonicalize!(ρ)
	normalize && lmul!(1 / norm(ρ), ρ)
	return ρ
end
randommpo(ds::AbstractVector{Int}; kwargs...) = randommpo(Float64, ds; kwargs...)

"""
	DensityOperator(ψ::CanonicalMPS) -> CanonicalMPO

The pure-state density matrix `ρ = |ψ⟩⟨ψ|` in operator (MPO) form: site tensors are the
per-site tensor products `ψᵢ ⊗ conj(ψᵢ)` with fused bond indices (bond dimensions squared).
"""
function DensityOperator(ψ::CanonicalMPS)
	L = length(ψ)
	T = scalartype(ψ)
	Ws = Vector{Array{T,4}}(undef, L)
	for i in 1:L
		A = ψ[i]
		@tensor W6[a, o1, c, o2, b, d] := A[a, o1, c] * conj(A[b, o2, d])
		W6p = permute(W6, (1, 5, 2, 3, 6, 4))
		Ws[i] = tie(W6p, (2, 1, 2, 1))
	end
	return CanonicalMPO(Ws; scaling=scaling(ψ)^2)
end

"""
	infinite_temperature_state(::Type{T}, ds) -> CanonicalMPO

The infinite-temperature state `I/2^{L/2}` as a `CanonicalMPO` (`tr(ρ) = 1`).
"""
function infinite_temperature_state(::Type{T}, ds::AbstractVector{Int}) where {T<:Number}
	ρ = CanonicalMPO(T, ds)
	setscaling!(ρ, prod(ds)^(-1 / length(ds)))
	return ρ
end

"""
	increase_bond!(ψ::CanonicalMPS; D) -> ψ

Grow the bond dimensions of `ψ` towards the profile `max_bonddims(ds, D)` by
zero padding (the state is unchanged), then re-canonicalize without truncation.
Used to seed the initial guess of `DMRG1`.
"""
function increase_bond!(ψ::CanonicalMPS; D::Int=Defaults.D)
	isempty(ψ.data) && return ψ
	T = scalartype(ψ)
	ds = phydims(ψ)
	L = length(ψ)
	Dl = ones(Int, L + 1)
	for i in 1:L
		Dl[i+1] = min(D, Dl[i] * ds[i])
	end
	Dr = ones(Int, L + 1)
	for i in L:-1:1
		Dr[i] = min(D, Dr[i+1] * ds[i])
	end
	b = min.(Dl, Dr)
	for i in 1:L-1   # never shrink below the current bond dimension
		b[i+1] = max(b[i+1], bonddim(ψ, i))
	end
	newdata = Vector{Array{T,3}}(undef, L)
	for i in 1:L
		dl = i == 1 ? 1 : b[i]
		dr = i == L ? 1 : b[i+1]
		t = zeros(T, dl, ds[i], dr)
		t[1:size(ψ[i], 1), :, 1:size(ψ[i], 3)] = ψ[i]
		newdata[i] = t
	end
	copy!(ψ.data, newdata)
	canonicalize!(ψ; alg=Orthogonalize(SVD(), NoTruncation(), false))
	return ψ
end

"""
	truncate!(ψ::CanonicalMPS; trunc=DefaultOrthTruncation) -> (ψ, err)

Truncate `ψ` by canonicalizing with an SVD sweep under the truncation scheme `trunc`.
Returns `ψ` and the maximal truncation error.
"""
truncate!(ψ::CanonicalMPS; trunc::TruncationScheme=DefaultOrthTruncation) =
	canonicalize!(ψ; alg=Orthogonalize(SVD(), trunc, false))
