using CUDArt: CudaArray
import Base.reshape
reshape(a::CudaArray, dims::Tuple) = CudaArray(a.ptr, dims, a.dev)
reshape(a::CudaArray, dims::Integer...) = reshape(a, dims)

import CUDArt.to_host
to_host{T <: AbstractFloat}(a::Array{T}) = copy(a)
