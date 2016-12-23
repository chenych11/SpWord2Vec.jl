import Base.BLAS: gemm!, gemv!, syr!

function takelast!{F <: AbstractFloat, T <: Integer}(
    dest::Array{F}, src::Array{F},
    idxes::Union{Tuple{Vararg{T}}, Array{T}})
    pre = ntuple(i->Colon(), ndims(src)-1)
    for (i, k) in enumerate(idxes)
        dest[pre..., i] = src[pre..., k]
    end
    dest
end

takelast!(dest, src, idx::Integer) = takelast!(dest, src, (idx,))

function viewlast{F <: AbstractFloat}(src::Array{F}, idx::Integer)
    pre = ntuple(i->Colon(), ndims(src)-1)
    @view src[pre..., idx]
end

function splitarray(a::Vector, shape1::Tuple, shape2::Tuple)
    l1 = prod(shape1)
    l2 = prod(shape2)
    @assert length(a) == l1 + l2
    p1 = @view a[1:l1]
    p2 = @view a[l1+1:end]
    return (reshape(p1, shape1), reshape(p2, shape2))
end

if BACKEND == "GPU"
    include("gpu_array.jl")
end
