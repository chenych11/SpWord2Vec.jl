using CUDArt: CudaArray, HostArray
using CUBLAS
using CUBLAS: statuscheck, cublasStatus_t, cublasHandle_t, cublasOperation_t, BlasChar, cublasop, cublashandle
using CUBLAS: gemm!, gemv!, syr!, gemm_batched!, axpy!

const libcublas = CUBLAS.libcublas

for (fname, elty) in
        ((:cublasDgemmStridedBatched, :Float64),
         (:cublasSgemmStridedBatched, :Float32),
         (:cublasZgemmStridedBatched, :Complex128),
         (:cublasCgemmStridedBatched, :Complex64))
    @eval begin
        #=
        cublasStatus_t  cublasDgemmStridedBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const double *alpha,
        const double *A, int lda, long long int strideA,
        const double *B, int ldb, long long int strideB,
        const double *beta,
        double *C, int ldc, long long int strideC, int batchCount);
        =#
        function gemm_strided_batched!(transA::BlasChar,
                                       transB::BlasChar,
                                       alpha::($elty),
                                       A::CudaArray{$elty, 3},
                                       B::CudaArray{$elty, 3},
                                       beta::($elty),
                                       C::CudaArray{$elty, 3})

            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)
            if m != size(C, 1) || n != size(C, 2) || k != size(B, transB == 'N' ? 1 : 2)
                throw(DimensionMismatch(""))
            end

            strideA = size(A, 1) * size(A, 2)
            strideB = size(B, 1) * size(B, 2)
            strideC = size(C, 1) * size(C, 2)
            batchCount = size(A, 3)

            cutransA = cublasop(transA)
            cutransB = cublasop(transB)
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))

            statuscheck(ccall(($(string(fname)), libcublas),
                cublasStatus_t,
                (cublasHandle_t, cublasOperation_t, cublasOperation_t,
                Cint, Cint, Cint, # m, n, k
                Ptr{$elty},       # alpha
                Ptr{$elty}, Cint, Clonglong, # A, lda, strideA
                Ptr{$elty}, Cint, Clonglong, # B, ldb, strideB
                Ptr{$elty},       # beta
                Ptr{$elty}, Cint, Clonglong, # C, ldc, strideC
                Cint),  # batchCount
                cublashandle[1], cutransA, cutransB,
                m, n, k,
                [alpha],
                A, lda, strideA,
                B, ldb, strideB,
                [beta],
                C, ldc, strideC,
                batchCount))
            C
        end
    end
end

"""
```
function takelast!{F <: AbstractFloat, T <: Integer}(
    dest::CudaArray{F}, src::CudaArray{F},
    idxes::Union{Tuple{Vararg{T}}, Array{T}})
```
get subarray by last dimension.
"""
function takelast!{F <: AbstractFloat, T <: Integer}(
    dest::CudaArray{F}, src::CudaArray{F},
    idxes::Union{Tuple{Vararg{T}}, Array{T}})

    siz = size(src)[1:end-1]
    strides_ = prod(siz) * sizeof(eltype(src))

    for (i, k) in enumerate(idxes)
        dest_ptr = dest.ptr + (i-1) * strides_
        src_ptr = src.ptr + (k-1) * strides_
        src_ = CudaArray(src_ptr, siz, src.dev)
        dest_ = CudaArray(dest_ptr, siz, dest.dev)
        copy!(dest_, src_)
    end
    dest
end

function viewlast{F <: AbstractFloat}(src::CudaArray{F}, idx::Integer)
    siz = size(src)[1:end-1]
    strides_ = prod(siz) * sizeof(eltype(src))

    src_ptr = src.ptr + (idx-1) * strides_
    dest = CudaArray(src_ptr, siz, src.dev)
    dest
end

function splitarray(a::CudaArray, shape1::Tuple, shape2::Tuple) :: NTuple{2, CudaArray}
    l1 = prod(shape1)
    l2 = prod(shape2)
    @assert length(a) == l1 + l2
    p1_ptr = a.ptr
    p2_ptr = p1_ptr + l1 * sizeof(eltype(a))
    p1 = CudaArray(p1_ptr, shape1, a.dev)
    p2 = CudaArray(p2_ptr, shape2, a.dev)
    return (p1, p2)
end
