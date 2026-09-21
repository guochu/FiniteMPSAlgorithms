# vendored from TEMPO/src/tensorops/distance.jl

function _distance2(x, y)
	sA = real(dot(x, x))
	sB = real(dot(y, y))
	c = dot(x, y)
	r = sA+sB-2*real(c)
	return abs(r)
end

_distance(x, y) = sqrt(_distance2(x, y))


"""
    distance2(x, y)

Compute the squared distance between two arrays (or tensor networks) of the same shape:

`‖x - y‖² = ‖x‖² + ‖y‖² - 2 Re⟨x, y⟩`; the absolute value is returned to avoid negative values from floating-point rounding.

Typically used to verify the accuracy of tensor network multiplication against a naive reference implementation. See also [`distance`](@ref).
"""
distance2(x::AbstractArray{<:Number, N}, y::AbstractArray{<:Number, N}) where N = _distance2(x, y)
"""
    distance(x, y)

The 2-norm distance between two arrays (or tensor networks) of the same shape, equal to `sqrt(distance2(x, y))`.
"""
distance(x::AbstractArray{<:Number, N}, y::AbstractArray{<:Number, N}) where N = _distance(x, y)
