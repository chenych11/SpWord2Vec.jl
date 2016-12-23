module SpSkipGramA
using CUDArt: HostArray, CudaArray, synchronize, null_stream
using JLD: jldopen
using CycUtils: invertable_unique, get, set!
using Veclib: gamma_kernel!, gamma_kernel_loss!, normalize_columns!

export SparseSkipGramA, train, init, close_device

include("config.jl")
include("array_handler.jl")
include("samples.jl")
include("utils.jl")

type SparseSkipGramA{Tensor} # Tensor could be HostArray{Float32, 2}
    α::Tensor
    β::Tensor
    B::Tensor
    C::Tensor

    # the following fields are training parameters.
    m::Int
    k::Int
    lr::Float32
    λ::Float32

    # maps
    idx2word::Array{String, 1}
    word2idx::Dict{String, UInt32}

    # the following fields are named memory for training
    ᾱ::Tensor
    β̄::Tensor
    a⃗::Tensor
    b⃗::Tensor
    γ::Tensor
    σ::Tensor
    ∇α::Tensor
    N::Tensor
    M::Tensor
    ∇β::Tensor
    Q::Tensor
    ∇B::Tensor
    ∇C::Tensor
    idxes::Tensor
    sign::Tensor

    function SparseSkipGramA(d::Integer, # vector dimension
        num_base::Integer,     # number of atom words
        num_vocab::Integer,    # number of vocabulary
        mini_batch::Integer,     # mini-batch size
        negative::Integer,        # number of negative samples per positive sample
        lr::AbstractFloat=0.01,       # learning rate
        λ::AbstractFloat=0.1,         # sparseness weight
        init_model::String=""   # dumped init_model.
    )
        @assert num_base ≤ num_vocab "The number of atom words must be less than the vocabulary size"

        α_ = randn(Float32, num_base, num_vocab) ./ sqrt(num_base)
        β_ = randn(Float32, num_base, num_vocab) ./ sqrt(num_base)
        B_ = rand(Float32, d, num_base)
        C_ = rand(Float32, d, num_base)

        i2w = Array{String, 1}()
        w2i = Dict{String, UInt32}()

        if init_model != ""
            e, c, i2w = jldopen(init_model, "r") do f
                (read(f, "embeddings"), read(f, "contexts"),
                 read(f, "idx2word"))
            end
            @assert num_vocab ≤ length(i2w) "Initial model must contains as many vocabularies as the model to be trained."
            @assert d == size(e, 1) "Initial vector dim. and specified dim. must agree."

            i2w = i2w[1:num_vocab]
            for (i, w) in enumerate(i2w)
                w2i[w] = i
            end

            α_[1:num_base, 1:num_base] = eye(num_base)
            β_[1:num_base, 1:num_base] = eye(num_base)
            copy!(B_, @view e[:, 1:num_base])
            copy!(C_, @view c[:, 1:num_base])
        end

        α = Tensor(α_)
        β = Tensor(β_)
        B = Tensor(B_)
        C = Tensor(C_)

        ᾱ = Tensor(Float32, num_base, mini_batch)
        β̄ = Tensor(Float32, num_base, negative+1, mini_batch)
        a⃗ = Tensor(Float32, d, mini_batch)
        b⃗ = Tensor(Float32, d, negative+1, mini_batch)
        γ = Tensor(Float32, negative+1, mini_batch)
        σ = Tensor(Float32, negative+1, mini_batch)
        ∇α = Tensor(Float32, num_base, mini_batch)
        N = Tensor(Float32, d, mini_batch)
        M = Tensor(Float32, num_base, mini_batch)
        ∇β = Tensor(Float32, num_base, negative+1, mini_batch)
        Q = Tensor(Float32, num_base, mini_batch)
        ∇B = Tensor(Float32, d, num_base)
        ∇C = Tensor(Float32, d, num_base)
        idxes = Tensor(UInt32, (negative+2)*mini_batch)

        fill!(ᾱ, 0.0)
        fill!(β̄, 0.0)
        fill!(a⃗, 0.0)
        fill!(b⃗, 0.0)
        fill!(γ, 0.0)
        fill!(σ, 0.0)
        fill!(∇α, 0.0)
        fill!(N, 0.0)
        fill!(M, 0.0)
        fill!(∇β, 0.0)
        fill!(Q, 0.0)
        fill!(∇B, 0.0)
        fill!(∇C, 0.0)
        fill!(idxes, 0)

        tmp = -ones(Float32, (negative+1, mini_batch))
        tmp[1, :] = 1.0f0
        s = Tensor(tmp)

        new(α, β, B, C,                               # model parameters
            mini_batch, negative, lr, λ,              # training parameters
            i2w, w2i,                                 # maps
            ᾱ, β̄, a⃗, b⃗, γ, σ, ∇α, N, M, ∇β, Q, ∇B, ∇C, idxes, s #work memory
        )
    end
end

function save(model::SparseSkipGramA{Array}, fname::String; save_maps=false)
    jldopen(fname, "w") do f
        write(f, "α", model.α)
        write(f, "β", model.β)
        write(f, "B", model.B)
        write(f, "C", model.C)

        # the following fields are training parameters.
        write(f, "m", model.m)
        write(f, "k", model.k)
        write(f, "lr", model.lr)
        write(f, "λ", model.λ)

        # maps
        if save_maps == true
            write(f, "idx2word", model.idx2word)
            write(f, "word2idx", model.word2idx)
        end
    end
end

const lr = Ref{Float32}(0.0f0)

num_vocab(model::SparseSkipGramA) = size(model.α, 2)
num_base(model::SparseSkipGramA) = size(model.α, 1)
num_vecdim(model::SparseSkipGramA) = size(model.B, 1)
num_minibatch(model::SparseSkipGramA) = size(model.a⃗, 2)
num_negative(model::SparseSkipGramA) = size(model.b⃗, 2) - 1
v̇() = lr[]

function to_device{M <: Union{SparseSkipGramA{Array},
                    SparseSkipGramA{HostArray}}}(model::M)
    d = num_vecdim(model)
    num_base_ = num_base(model)
    num_vocab_ = num_vocab(model)
    mini_batch = num_minibatch(model)
    negative = num_negative(model)

    d_model = SparseSkipGramA{CudaArray}(d, num_base_, num_vocab_, mini_batch,
                                        negative, model.lr, model.λ)
    for fn in fieldnames(typeof(d_model))
        dest = getfield(d_model, fn)
        if isa(dest, CudaArray)
             src = getfield(model, fn)
             copy!(dest, src)
        end
    end
    return d_model
end

@generated function get_training_model(model::SparseSkipGramA)
    if BACKEND == "GPU"
        return :(to_device(model))
    else
        return :(model)
    end
end

# TODO: accelerate with multiple streams.
function compute_grad_sparse(model, batch)
    const d = num_vecdim(model)
    const m = model.m
    const k = model.k
    const B = model.B
    const C = model.C
    const N = model.N
    const M = model.M
    w, c = batch
    const nb = num_base(model)
    const ᾱ = takelast!(model.ᾱ, model.α, w)
    const β̄ = takelast!(model.β̄, model.β, c)
    const a⃗ = model.a⃗
    const b⃗ = model.b⃗
    const γ = model.γ
    const ∇α = model.∇α
    const ∇β = model.∇β

    flat_b⃗ = reshape(b⃗, (d, (k+1)*m))
    gemm!('N', 'N', 1.0f0, B, ᾱ, 0.0f0, a⃗)
    gemm!('N', 'N', 1.0f0, C, reshape(β̄, (nb, (k+1)*m)), 0.0f0, flat_b⃗)

    # TODO: for GPU, we can use multiple of streams to overlap the computations and data transferring.  This need profiling. make a benchmark test first. Candidate solutions: (1) simple iteration with one stream; (2) simple iteration with multiple streams; (3) batched calling with gemm.

    a⃗_ = reshape(a⃗, (d, 1, m))
    γ_ = reshape(γ, (k+1, 1, m))
    gemm_strided_batched!('T', 'N', 1.0f0, b⃗, a⃗_, 0.0f0, γ_)
    gamma_kernel_loss!(γ, model.σ, model.sign)

    # ∇α
    N_ = reshape(N, (d, 1, m))
    gemm_strided_batched!('N', 'N', 1.0f0, b⃗, γ_, 0.0f0, N_)
    gemm!('T', 'N', 1.0f0, B, N, 0.0f0, ∇α)

    # ∇β
    gemm!('T', 'N', 1.0f0, C, a⃗, 0.0f0, M)
    M_ = reshape(M, (num_base(model), 1, m))
    gemm_strided_batched!('N', 'T', 1.0f0, M_, γ_, 0.0f0, ∇β)
    nothing
end


function accumulate_grad_dict(model, t)
    const m = model.m
    const k = model.k
    w1 = (Float32(t)/(t+1)) :: Float32
    w2 = (1.0f0/(t+1)) :: Float32

    gemm!('N', 'T', w2, model.N, model.ᾱ, w1, model.∇B)
    β̄ = model.β̄  # |B|×(k+1)×m
    γ = reshape(model.γ, (k+1, 1, m))  # (k+1) × 1 × m
    Q = reshape(model.Q, (num_base(model), 1, m)) # |B| × 1 × m
    gemm_strided_batched!('N', 'N', 1.0f0, β̄, γ, 0.0f0, Q)
    gemm!('N', 'T', w2, model.a⃗, model.Q, w1, model.∇C)
    nothing
end

function fetch_grad_sparse!(grad::Tuple, model)
    copy!(grad[1], model.∇α)
    copy!(grad[2], model.∇β)
    return grad
end

# function cpu_sparse_update!(arr, grad, idxes, lr, lambda)
#     for (j, idx) in zip(1:size(grad,2), idxes)
#         @inbounds for i in 1:size(arr, 1)
#             x = arr[i, idx] -  lr * grad[i, j]
#             s = (x > 0.0f0 ? 1.0f0 : -1.0f0)
#             x *= s
#             x -= lambda*lr
#             x = x > 0.0f0 ? x : 0.0f0
#             x = x < 1.0f0 ? x : 1.0f0
#             arr[i, idx] = x * s
#         end
#     end
# end

function gpu_sparse_update!(arr, grad, idxes, lr, lambda)
    # make sure the transferring completes.
    CUDArt.synchronize(CUDArt.null_stream)
    ccall((:sparse_update, vec_lib), Void, (Ptr{Float32}, Ptr{Float32},
        Cint, Cint, Ptr{Cuint}, Float32, Float32, Ptr{Void}, Cint),
        arr.ptr, grad.ptr, size(grad, 1), size(grad, 2),
        idxes.ptr, lr, lambda, null_stream.inner.handle, 256)
end

@generated function proximal_update!(model, grads, idxes)
    if BACKEND == "GPU"
        quote
            global lr
            lr_::Float32 = lr[]
            dev = Int(DEVICE)
            ∇α, ∇β = grads
            w, c = idxes
            siz_α = (num_base(model), length(w))
            siz_β = (num_base(model), length(c))
            grad_α = CudaArray(model.∇α.ptr, siz_α, dev)
            grad_β = CudaArray(model.∇β.ptr, siz_β, dev)
            copy!(grad_α, ∇α)
            copy!(grad_β, ∇β)

            ptr_w = model.idxes.ptr
            ptr_c = ptr_w + length(w)*sizeof(UInt32)
            d_w = CudaArray(ptr_w, (length(w),), dev)
            d_c = CudaArray(ptr_c, (length(c),), dev)
            copy!(d_c, c)
            copy!(d_w, w)
            # CUDArt.synchronize(CUDArt.null_stream)
            gpu_sparse_update!(model.α, grad_α, d_w, lr_, model.λ)
            gpu_sparse_update!(model.β, grad_β, d_c, lr_, model.λ)
        end
    else
        quote
            error("Not implemented")
            # global lr
            # lr_::Float32 = lr[]
            # ∇α, ∇β = grads
            # w, c = idxes
            #
            # cpu_sparse_update!(model.α, ∇α, w, lr_, model.λ)
            # cpu_sparse_update!(model.β, ∇β, c, lr_, model.λ)
        end
    end
end

function update_sparse_u1!(model, grad_cache, grad, uniq_batch)
    ∇α, ∇β = grad_cache
    grad_α, grad_β = grad
    uniq_w, uniq_c = uniq_batch
    w, inv_w = uniq_w
    c, inv_c = uniq_c

    ∇α = @view ∇α[:, 1:length(w)]
    ∇β = @view ∇β[:, 1:length(c)]
    fill!(∇α, 0.0f0)
    fill!(∇β, 0.0f0)

    # make sure the transferring completes.
    CUDArt.synchronize(CUDArt.null_stream)

    for (i, inv_idx) in enumerate(inv_w)
        ∇α[:, inv_idx] += grad_α[:, i]
    end

    for (i, inv_idx) in enumerate(inv_c)
        ∇β[:, inv_idx] += grad_β[:, i]
    end
    proximal_update!(model, [∇α, ∇β], [w, c])
    nothing
end

function update_sparse_u2!(model, grad_cache, grad, uniq_batch)
    ∇α, ∇β = grad_cache
    grad_α, grad_β = grad
    uniq_w, uniq_c = uniq_batch
    w, inv_w = uniq_w
    c, inv_c = uniq_c

    ∇α = @view ∇α[:, 1:length(w)]
    ∇β = @view ∇β[:, 1:length(c)]
    fill!(∇α, 0.0f0)
    fill!(∇β, 0.0f0)

    # make sure the transferring completes.
    CUDArt.synchronize(CUDArt.null_stream)

    for (i, inv_idx) in enumerate(inv_w)
        ∇α[:, inv_idx] += grad_α[:, i]
    end
    normalize_columns!(∇α)

    for (i, inv_idx) in enumerate(inv_c)
        ∇β[:, inv_idx] += grad_β[:, i]
    end
    normalize_columns!(∇β)

    proximal_update!(model, [∇α, ∇β], [w, c])
    nothing
end

function update_dict_u1!(model)
    global lr
    lr_::Float32 = lr[]
    axpy!(-lr_, vec(model.∇B), vec(model.B))
    axpy!(-lr_, vec(model.∇C), vec(model.C))
    nothing
end

const ∇B_cache = Ref{HostArray}()
const ∇C_cache = Ref{HostArray}()
function update_dict_u2!(model)
    global ∇B_cache
    global ∇C_cache
    ∇B = ∇B_cache[]
    ∇C = ∇C_cache[]
    lr_::Float32 = lr[]

    copy!(∇B, model.∇B)
    CUDArt.synchronize(CUDArt.null_stream)
    normalize_columns!(∇B)
    copy!(model.∇B, ∇B)
    copy!(∇C, model.∇C)
    CUDArt.synchronize(CUDArt.null_stream)
    axpy!(-lr_, vec(model.∇B), vec(model.B))
    normalize_columns!(∇C)
    copy!(model.∇C, ∇C)
    axpy!(-lr_, vec(model.∇C), vec(model.C))
    nothing
end

function update_all_u11!(model, grad_cache, grad, uniq_batch)
    update_sparse_u1!(model, grad_cache, grad, uniq_batch)
    update_dict_u1!(model)
end

function update_all_u12!(model, grad_cache, grad, uniq_batch)
    update_sparse_u1!(model, grad_cache, grad, uniq_batch)
    update_dict_u2!(model)
end

function update_all_u21!(model, grad_cache, grad, uniq_batch)
    update_sparse_u2!(model, grad_cache, grad, uniq_batch)
    update_dict_u1!(model)
end

function update_all_u22!(model, grad_cache, grad, uniq_batch)
    update_sparse_u2!(model, grad_cache, grad, uniq_batch)
    update_dict_u2!(model)
end

@generated function sync_params!(cpu_model, model)
    if BACKEND == "CPU"
        return :(cpu_model)
    else
        quote
            for fn in (:α, :β, :B, :C)  #fieldnames(typeof(model))
                src = getfield(model, fn)
                if isa(src, CudaArray)
                     dest = getfield(cpu_model, fn)
                     copy!(dest, src)
                end
            end
            CUDArt.synchronize(CUDArt.null_stream)
            cpu_model
        end
    end
end

const _start_time_loginfo = Ref{Float64}(0.0)
const _next_report_time = Ref{Float64}(0.0)
const _sigmas_loginfo = Ref{HostArray{Float32, 2}}()

function log_progress(model, current, progress=progress)
    global _start_time_loginfo
    global _next_report_time
    global lr
    global _sigmas_loginfo

    if current >= _next_report_time[]
        σ = _sigmas_loginfo[]
        copy!(σ, model.σ)
        loss = 0.0f0
        for x in σ
            loss += -log(x+eps(typeof(x)))
        end
        ppl = exp(loss/(model.m*(model.k+1)))

        _next_report_time[] += 3.0
        st = _start_time_loginfo[]
        factory = get(sampleFactory)
        prog = progress(factory) * 100.0
        elapsed = current - st
        etc = (100.0 - prog) / prog * elapsed
        hours = div(etc, 3600)
        rem = etc-hours*3600
        mins = div(rem, 60)
        secs = rem - mins * 60

        mean_speed = factory.consumed[]/elapsed
        info(@sprintf("ETA: %02d:%02d:%02d - loss: %.4f - ppl: %.3f - progress: %5.2f%% - speed: %.1f lines/s - lr: %g", hours, mins, secs, loss, ppl, prog, mean_speed, lr[]))
    end
    nothing
end

function init_logger(model, start_time)
    global _start_time_loginfo
    global _next_report_time
    global _sigmas_loginfo
    _start_time_loginfo[] = start_time
    _next_report_time[] = start_time
    _sigmas_loginfo[] = HostArray(Float32, (model.k+1), model.m)
end

function train(model; min_lr=1f-4, normalize_sp=false,
        normalize_dict=false, every=500, preratio=0.1,
        saveratio=0.05, part=1.0, save_basename="A")
    global sampleFactory
    global lr

    cpu_model = model
    model = to_device(cpu_model)

    beta = model.lr / min_lr - 1.0f0
    lr[] = model.lr

    info(@sprintf("Train with lr=%g, normalize_sp=%s, normalize_dict=%s. Update dictionary parameters every %d updates of sparse parameters. Pretrain sparse parameters with %.2f%% amount of training data. Save after every %.2f%% of training. Use %.2f%% of the total training data. Save the trained model to \"%s*.jld\"", lr[], string(normalize_sp), string(normalize_dict), every, preratio*100, saveratio*100, part*100, save_basename))

    st = time()
    init_logger(model, st)

    grads = (HostArray(Float32, num_base(model), model.m),
             HostArray(Float32, num_base(model), (model.k+1), model.m))
    grad_cache = (HostArray(Float32, num_base(model), model.m),
            HostArray(Float32, num_base(model), (model.k+1)*model.m))

    factory = get(sampleFactory)
    const producer = factory.producer
    batch_set = Task(() -> producer(factory, model.m, model.k))

    const update_sparse! = normalize_sp ? update_sparse_u2! : update_sparse_u1!
    if !normalize_dict && !normalize_sp
        const update_all! = update_all_u11!
    elseif normalize_dict && !normalize_sp
        const update_all! = update_all_u12!
    elseif normalize_sp && !normalize_dict
        const update_all! = update_all_u21!
    else
        const update_all! = update_all_u22!
    end
    const progress_ = factory -> progress(factory)/part

    batch = consume(batch_set)
    old_batch = batch
    compute_grad_sparse(model, old_batch) # this method immediately returns when
                                  # the computations are taking place on a GPU.
                                  # this makes the gpu and cpu runs concurrently
    for batch in batch_set
        uniq_w = invertable_unique(old_batch[1])
        uniq_c = invertable_unique(old_batch[2])
        uniq_batch = (uniq_w, uniq_c)
        # wait for gpu computing completes
        # synchronize(CUDArt.null_stream)
        fetch_grad_sparse!(grads, model)
        update_sparse!(model, grad_cache, grads, uniq_batch)

        old_batch = batch
        prog = progress_(factory)
        if prog >= preratio
            info("Pre-train finished.")
            break
        end
        compute_grad_sparse(model, old_batch)
        log_progress(model, time(), progress_)
        lr[] = model.lr / (1.0f0 + beta*prog)
    end

    next_save_point = preratio + saveratio
    compute_grad_sparse(model, old_batch)
    nb_batch = 0
    for batch in batch_set
        uniq_w = invertable_unique(old_batch[1])
        uniq_c = invertable_unique(old_batch[2])
        uniq_batch = (uniq_w, uniq_c)
        # TODO: maybe another stream to handle the data transferring
        fetch_grad_sparse!(grads, model)
        accumulate_grad_dict(model, nb_batch)
        if nb_batch == every - 1
            # compute_grad_BC(model)
            update_all!(model, grad_cache, grads, uniq_batch)
        else
            update_sparse!(model, grad_cache, grads, uniq_batch)
        end

        old_batch = batch
        compute_grad_sparse(model, old_batch)
        prog = progress_(factory)

        if prog >= next_save_point
            if prog >= 1.0
                break
            end
            # synchronize
            sync_params!(cpu_model, model)
            sn = joinpath(project_root,
                @sprintf("%s-checkpoint-%03.0f.jld", save_basename, prog*100.0f0))
            info("Save point reached. saveing to $sn")
            save(cpu_model, sn)
            next_save_point += saveratio
        end
        log_progress(model, time(), progress_)
        lr[] = model.lr / (1.0f0 + beta*prog)
        nb_batch = (nb_batch+1) % every
    end

    sync_params!(cpu_model, model)
    sn = joinpath(project_root,
        @sprintf("%s-checkpoint-100.jld", save_basename))
    info("Training finished. Saving to $sn")
    save(cpu_model, sn, save_maps=true)
end

end # end of module
