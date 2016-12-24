module SpSkipGramC
# share sparse representations
using CUDArt: HostArray, CudaArray, synchronize, null_stream
using JLD: jldopen
using CycUtils: invertable_unique, get, set!
using Veclib: gamma_kernel!, gamma_kernel_loss!, normalize_columns!

export SparseSkipGramC, train, init, close_device

include("config.jl")
include("array_handler.jl")
include("samples.jl")
include("utils.jl")

type SparseSkipGramC{Tensor} # Tensor could be HostArray{Float32, 2}
    α::Tensor
    B::Tensor
    V::Tensor

    # the following fields are training parameters.
    m::Int
    k::Int
    lr::Float32
    lr0::Float32
    λ::Float32

    # maps
    idx2word::Array{String, 1}
    word2idx::Dict{String, UInt32}

    # the following fields are named memory for training
    ᾱ::Tensor
    a⃗::Tensor
    b⃗::Tensor
    γ::Tensor
    σ::Tensor
    ∇α::Tensor
    N::Tensor
    ∇B::Tensor
    ∇V::Tensor
    idxes::Tensor
    sign::Tensor

    function SparseSkipGramC(d::Integer, # vector dimension
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
        B_ = rand(Float32, d, num_base)
        V_ = rand(Float32, d, num_vocab)

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
            copy!(B_, @view e[:, 1:num_base])
            copy!(V_, @view c[:, 1:num_vocab])
        end

        α = Tensor(α_)
        B = Tensor(B_)
        V = Tensor(V_)

        ᾱ = Tensor(Float32, num_base, mini_batch)
        a⃗ = Tensor(Float32, d, mini_batch)
        b⃗ = Tensor(Float32, d, negative+1, mini_batch)
        γ = Tensor(Float32, negative+1, mini_batch)
        σ = Tensor(Float32, negative+1, mini_batch)
        ∇α = Tensor(Float32, num_base, mini_batch)
        N = Tensor(Float32, d, mini_batch)
        ∇B = Tensor(Float32, d, num_base)
        ∇V = Tensor(Float32, d, negative+1, mini_batch)
        idxes = Tensor(UInt32, (negative+2)*mini_batch)

        fill!(ᾱ, 0.0)
        fill!(a⃗, 0.0)
        fill!(b⃗, 0.0)
        fill!(γ, 0.0)
        fill!(σ, 0.0)
        fill!(∇α, 0.0)
        fill!(N, 0.0)
        fill!(∇B, 0.0)
        fill!(∇V, 0.0)
        fill!(idxes, 0)

        tmp = -ones(Float32, (negative+1, mini_batch))
        tmp[1, :] = 1.0f0
        s = Tensor(tmp)

        new{Tensor}(α, B, V, # model parameters
          mini_batch, negative, lr, lr, λ, # training parameters
          i2w, w2i,  # maps
          ᾱ, a⃗, b⃗, γ, σ, ∇α, N, ∇B, ∇V, idxes, s
        )
    end
end

function save(model::SparseSkipGramC{Array}, fname::String; save_maps=false)
    jldopen(fname, "w") do f
        write(f, "α", model.α)
        write(f, "B", model.B)
        write(f, "V", model.V)

        # the following fields are training parameters.
        write(f, "m", model.m)
        write(f, "k", model.k)
        write(f, "lr", model.lr0)
        write(f, "λ", model.λ)

        # maps
        if save_maps == true
            write(f, "idx2word", model.idx2word)
            write(f, "word2idx", model.word2idx)
        end
    end
end

num_vocab(model::SparseSkipGramC) = size(model.α, 2)
num_base(model::SparseSkipGramC) = size(model.α, 1)
num_vecdim(model::SparseSkipGramC) = size(model.B, 1)
num_minibatch(model::SparseSkipGramC) = size(model.a⃗, 2)
num_negative(model::SparseSkipGramC) = size(model.b⃗, 2) - 1

function to_device{M <: Union{SparseSkipGramC{Array},
                    SparseSkipGramC{HostArray}}}(model::M)
    d = num_vecdim(model)
    num_base_ = num_base(model)
    num_vocab_ = num_vocab(model)
    mini_batch = num_minibatch(model)
    negative = num_negative(model)

    d_model = SparseSkipGramC{CudaArray}(d, num_base_, num_vocab_, mini_batch,
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

@generated function get_training_model(model::SparseSkipGramC)
    if BACKEND == "GPU"
        return :(to_device(model))
    else
        return :(model)
    end
end

function compute_grad_sparse(model, batch)
    const d = num_vecdim(model)
    const m = model.m
    const k = model.k
    const B = model.B
    const V = model.V
    const N = model.N
    w, c = batch
    const nb = num_base(model)
    const ᾱ = takelast!(model.ᾱ, model.α, w)
    const a⃗ = model.a⃗
    const b⃗ = takelast!(model.b⃗, model.V, c)
    const γ = model.γ
    const ∇α = model.∇α
    const ∇V = model.∇V
    const ∇B = model.∇B

    # flat_b⃗ = reshape(b⃗, (d, (k+1)*m))
    gemm!('N', 'N', 1.0f0, B, ᾱ, 0.0f0, a⃗)

    a⃗_ = reshape(a⃗, (d, 1, m))
    γ_ = reshape(γ, (k+1, 1, m))
    gemm_strided_batched!('T', 'N', 1.0f0, b⃗, a⃗_, 0.0f0, γ_)
    gamma_kernel_loss!(γ, model.σ, model.sign)

    #∇α
    N_ = reshape(N, (d, 1, m))
    gemm_strided_batched!('N', 'N', 1.0f0, b⃗, γ_, 0.0f0, N_)
    gemm!('T', 'N', 1.0f0, B, N, 0.0f0, ∇α)

    # ∇V
    gemm_strided_batched!('N', 'T', 1.0f0, a⃗_, γ_, 0.0f0, ∇V)

    nothing
end

function accumulate_grad_dict(model, t)
    const m = model.m
    const k = model.k

    w1 = (Float32(t)/(t+1)) :: Float32
    w2 = (1.0f0/(t+1)) :: Float32

    # ∇B
    # gemm!('N', 'T', 1.0f0, N, ᾱ, 0.0f0, ∇B)
    gemm!('N', 'T', w2, model.N, model.ᾱ, w1, model.∇B)
    nothing
end

function fetch_grad_sparse!(grads, model)
    copy!(grads[1], model.∇α)
    copy!(grads[2], model.∇V)
    return grads
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
    ccall((:sparse_update, vec_lib), Void, (Ptr{Float32}, Ptr{Float32},
        Cint, Cint, Ptr{Cuint}, Float32, Float32, Ptr{Void}, Cint),
        arr.ptr, grad.ptr, size(grad, 1), size(grad, 2),
        idxes.ptr, lr, lambda, null_stream.inner.handle, 256)
end

function gpu_V_update!(arr, grad, idxes, lr)
    ccall((:sparse_update_V, vec_lib), Void, (Ptr{Float32}, Ptr{Float32},
        Cint, Cint, Ptr{Cuint}, Float32, Ptr{Void}, Cint),
        arr.ptr, grad.ptr, size(grad, 1), size(grad, 2),
        idxes.ptr, lr, null_stream.inner.handle, 256)
end

@generated function proximal_update!(model, grads::Tuple, idxes::Tuple)
    if BACKEND == "GPU"
        quote
            lr_::Float32 = model.lr
            dev = Int(DEVICE)
            ∇α, ∇V = grads
            w, c = idxes
            siz_α = (num_base(model), length(w))
            siz_V = (num_vecdim(model), length(c))
            grad_α = CudaArray(model.∇α.ptr, siz_α, dev)
            grad_V = CudaArray(model.∇V.ptr, siz_V, dev)
            copy!(grad_α, ∇α)
            copy!(grad_V, ∇V)

            ptr_w = model.idxes.ptr
            ptr_c = ptr_w + length(w)*sizeof(UInt32)
            d_w = CudaArray(ptr_w, (length(w),), dev)
            d_c = CudaArray(ptr_c, (length(c),), dev)
            copy!(d_c, c)
            copy!(d_w, w)

            # make sure the transferring completes.
            CUDArt.synchronize(CUDArt.null_stream)
            gpu_sparse_update!(model.α, grad_α, d_w, lr_, model.λ)
            gpu_V_update!(model.V, grad_V, d_c, lr_)
        end
    else
        quote
            error("Not implemented")
            # lr_::Float32 = model.lr
            # ∇α, ∇β = grads
            # w, c = idxes
            #
            # cpu_sparse_update!(model.α, ∇α, w, lr_, model.λ)
            # cpu_V_update!(model.V, grad_V, d_c, lr_)
        end
    end
end


function update_sparse_u2!(model, grad_cache::Tuple, grad_αV::Tuple, uniq_batch)
    const δ = inv(prevfloat(typemax(Float32)))
    uniq_w, uniq_c = uniq_batch
    idx_w, inv_w = uniq_w
    idx_c, inv_c = uniq_c

    ∇α, ∇V = grad_cache
    grad_α, grad_V = grad_αV
    grad_V = reshape(grad_V, (size(grad_V, 1), size(grad_V, 2)*size(grad_V, 3)))

    ∇α = @view ∇α[:, 1:length(idx_w)]
    ∇V = @view ∇V[:, 1:length(idx_c)]
    fill!(∇α, 0.0f0)
    fill!(∇V, 0.0f0)

    # make sure the transferring completes.
    CUDArt.synchronize(CUDArt.null_stream)

    @inbounds for (i, inv_idx) in enumerate(inv_w)
        ∇α[:, inv_idx] += grad_α[:, i]
    end
    normalize_columns!(∇α)

    @inbounds for (i, inv_idx) in enumerate(inv_c)
        ∇V[:, inv_idx] += grad_V[:, i]
    end
    normalize_columns!(∇V)

    proximal_update!(model, (∇α, ∇V), (idx_w, idx_c))
    nothing
end

function update_sparse_u1!(model, grad_cache::Tuple, grad_αV::Tuple, uniq_batch)
    const δ = inv(prevfloat(typemax(Float32)))
    uniq_w, uniq_c = uniq_batch
    idx_w, inv_w = uniq_w
    idx_c, inv_c = uniq_c

    ∇α, ∇V = grad_cache
    grad_α, grad_V = grad_αV
    grad_V = reshape(grad_V, (size(grad_V, 1), size(grad_V, 2)*size(grad_V, 3)))

    ∇α = @view ∇α[:, 1:length(idx_w)]
    ∇V = @view ∇V[:, 1:length(idx_c)]
    fill!(∇α, 0.0f0)
    fill!(∇V, 0.0f0)

    # make sure the transferring completes.
    CUDArt.synchronize(CUDArt.null_stream)

    @inbounds for (i, inv_idx) in enumerate(inv_w)
        ∇α[:, inv_idx] += grad_α[:, i]
    end

    @inbounds for (i, inv_idx) in enumerate(inv_c)
        ∇V[:, inv_idx] += grad_V[:, i]
    end

    proximal_update!(model, (∇α, ∇V), (idx_w, idx_c))
    nothing
end


const ∇B_cache = Ref{HostArray}()
function update_dict_u2!(model)
    global ∇B_cache
    ∇B = ∇B_cache[]
    const lr_::Float32 = model.lr

    copy!(∇B, model.∇B)
    CUDArt.synchronize(CUDArt.null_stream)
    normalize_columns!(∇B)
    copy!(model.∇B, ∇B)
    CUDArt.synchronize(CUDArt.null_stream)
    axpy!(-lr_, vec(model.∇B), vec(model.B))
    nothing
end

function update_dict_u1!(model)
    const lr_::Float32 = model.lr
    axpy!(-lr_, vec(model.∇B), vec(model.B))
    nothing
end

function update_dict_u1f1!(model)
    const m = model.m
    const k = model.k
    # ∇B
    # gemm!('N', 'T', 1.0f0, N, ᾱ, 0.0f0, ∇B)
    gemm!('N', 'T', -model.lr, model.N, model.ᾱ, 1.0f0, model.B)
    nothing
end

function update_dict_u2f1!(model)
    const m = model.m
    const k = model.k
    global ∇B_cache
    ∇B = ∇B_cache[]
    # ∇B
    gemm!('N', 'T', 1.0f0, N, ᾱ, 0.0f0, ∇B)
    copy!(∇B, model.∇B)
    CUDArt.synchronize(CUDArt.null_stream)
    normalize_columns!(∇B)
    copy!(model.∇B, ∇B)
    CUDArt.synchronize(CUDArt.null_stream)
    axpy!(-model.lr, vec(model.∇B), vec(model.B))
    nothing
end

@generated function sync_params!(cpu_model, model)
    if BACKEND == "CPU"
        return :(cpu_model)
    else
        quote
            for fn in (:α, :B, :V)
                src = getfield(model, fn)
                dest = getfield(cpu_model, fn)
                copy!(dest, src)
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
    global _sigmas_loginfo
    global sampleFactory
    factory = get(sampleFactory)

    if current >= _next_report_time[]
        σ = _sigmas_loginfo[]
        copy!(σ, model.σ)
        CUDArt.synchronize(CUDArt.null_stream)
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
        info(@sprintf("ETA: %02d:%02d:%02d - loss: %.4f - ppl: %.3f - progress: %5.2f%% - speed: %.1f %s/s - lr: %g", hours, mins, secs, loss, ppl, prog, mean_speed, factory.corpus_type == String ? "lines" : "bytes", model.lr))
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
        saveratio=0.05, part=1.0, save_basename="C")
    if every == 1
        train_f1(model, min_lr=min_lr, normalize_sp=normalize_sp,
        normalize_dict=normalize_dict, preratio=preratio,
        saveratio=saveratio, part=part, save_basename=save_basename)
        return nothing
    end
    global sampleFactory
    if normalize_dict
        global ∇B_cache
        ∇B_cache[] = HostArray(Float32, (num_vecdim(model), num_base(model)))
    end

    cpu_model = model
    model = to_device(cpu_model)

    beta = model.lr0 / min_lr - 1.0f0

    info(@sprintf("Train with lr=%g, normalize_sp=%s, normalize_dict=%s. Update dictionary parameters every %d updates of sparse parameters. Pretrain sparse parameters with %.2f%% amount of training data. Save after every %.2f%% of training. Use %.2f%% of the total training data. Save the trained model to \"%s*.jld\"", model.lr, string(normalize_sp), string(normalize_dict), every, preratio*100, saveratio*100, part*100, save_basename))

    st = time()
    init_logger(model, st)

    grads = (HostArray(Float32, num_base(model), model.m),
             HostArray(Float32, num_vecdim(model), (model.k+1), model.m))

    grad_cache = (HostArray(Float32, num_base(model), model.m),
                  HostArray(Float32, num_vecdim(model), (model.k+1)*model.m))

    factory = get(sampleFactory)
    const producer = factory.producer
    const progress_ = factory -> progress(factory)/part
    batch_set = Task(() -> producer(factory, model.m, model.k))

    const update_sparse! = normalize_sp ? update_sparse_u2! : update_sparse_u1!
    const update_dict! = normalize_dict ? update_dict_u2! : update_dict_u1!

    batch = consume(batch_set)
    old_batch = batch
    compute_grad_sparse(model, old_batch) # this method immediately returns when
                                  # the computations are taking place on a GPU.
                                  # this makes the gpu and cpu runs concurrently
    for batch in batch_set
        uniq_w = invertable_unique(old_batch[1])
        uniq_c = invertable_unique(old_batch[2])
        uniq_batch = (uniq_w, uniq_c)

        fetch_grad_sparse!(grads, model)
        update_sparse!(model, grad_cache, grads, uniq_batch)

        old_batch = batch
        prog = progress_(factory)
        if prog >= preratio
            if preratio > 0.0f0
                info("Pre-train finished.")
            end
            break
        end
        compute_grad_sparse(model, old_batch)
        log_progress(model, time(), progress_)
        model.lr = model.lr0 / (1.0f0 + beta*prog)
    end

    next_save_point = preratio + saveratio
    compute_grad_sparse(model, old_batch)
    nb_batch = 0
    for batch in batch_set
        uniq_w = invertable_unique(old_batch[1])
        uniq_c = invertable_unique(old_batch[2])
        uniq_batch = (uniq_w, uniq_c)
        # synchronize(CUDArt.null_stream)
        # TODO: maybe another stream to handle the data transferring
        fetch_grad_sparse!(grads, model)

        accumulate_grad_dict(model, nb_batch)
        if nb_batch == every - 1
            update_dict!(model)
        end
        update_sparse!(model, grad_cache, grads, uniq_batch)

        old_batch = batch
        compute_grad_sparse(model, old_batch)
        prog = progress_(factory)
        if prog >= next_save_point
            if prog >= 1.0f0
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
        model.lr = model.lr0 / (1.0f0 + beta*prog)
        nb_batch = (nb_batch+1) % every
    end

    sync_params!(cpu_model, model)
    sn = joinpath(project_root,
        @sprintf("%s-checkpoint-100.jld", save_basename))
    info("Training finished. Saving to $sn")
    save(cpu_model, sn, save_maps=true)
    nothing
end

function train_f1(model; min_lr=1f-4, normalize_sp=false,
        normalize_dict=false, preratio=0.1,
        saveratio=0.05, part=1.0, save_basename="B")
    global sampleFactory
    if normalize_dict
        global ∇B_cache
        ∇B_cache[] = HostArray(Float32, (num_vecdim(model), num_base(model)))
    end

    cpu_model = model
    model = to_device(cpu_model)

    beta = model.lr0 / min_lr - 1.0f0

    info(@sprintf("Model [C]. train_f1: Train with lr=%g, normalize_sp=%s, normalize_dict=%s. Update dictionary parameters every 1 updates of sparse parameters. Pretrain sparse parameters with %.2f%% amount of training data. Save after every %.2f%% of training. Use %.2f%% of the total training data. Save the trained model to \"%s*.jld\"", model.lr, string(normalize_sp), string(normalize_dict), preratio*100, saveratio*100, part*100, save_basename))

    st = time()
    init_logger(model, st)

    grads = (HostArray(Float32, num_base(model), model.m),
             HostArray(Float32, num_vecdim(model), (model.k+1), model.m))

    grad_cache = (HostArray(Float32, num_base(model), model.m),
                  HostArray(Float32, num_vecdim(model), (model.k+1)*model.m))

    factory = get(sampleFactory)
    const producer = factory.producer
    batch_set = Task(() -> producer(factory, model.m, model.k))

    const update_sparse! = normalize_sp ? update_sparse_u2! : update_sparse_u1!
    const update_dict! = normalize_dict ? update_dict_u2f1! : update_dict_u1f1!
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
        fetch_grad_sparse!(grads, model)
        update_sparse!(model, grad_cache, grads, uniq_batch)

        old_batch = batch
        prog = progress_(factory)
        if prog >= preratio
            if preratio > 0.0f0
                info("Pre-train finished.")
            end
            break
        end
        compute_grad_sparse(model, old_batch)
        log_progress(model, time(), progress_)
        model.lr = model.lr0 / (1.0f0 + beta*prog)
    end

    next_save_point = preratio + saveratio
    compute_grad_sparse(model, old_batch)
    for batch in batch_set
        uniq_w = invertable_unique(old_batch[1])
        uniq_c = invertable_unique(old_batch[2])
        uniq_batch = (uniq_w, uniq_c)
        # synchronize(CUDArt.null_stream)
        # TODO: maybe another stream to handle the data transferring
        fetch_grad_sparse!(grads, model)
        update_dict!(model)
        update_sparse!(model, grad_cache, grads, uniq_batch)

        old_batch = batch
        compute_grad_sparse(model, old_batch)
        prog = progress_(factory)
        if prog >= next_save_point
            if prog >= 1.0f0
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
        model.lr = model.lr0 / (1.0f0 + beta*prog)
    end

    sync_params!(cpu_model, model)
    sn = joinpath(project_root,
        @sprintf("%s-checkpoint-100.jld", save_basename))
    info("Training finished. Saving to $sn")
    save(cpu_model, sn, save_maps=true)
    nothing
end


end # end of module
