module SpSkipGramD
# share sparse representations

using CUDArt: HostArray, CudaArray, synchronize, null_stream
using JLD: jldopen
using CycUtils: invertable_unique, get, set!
using Veclib: gamma_kernel!, gamma_kernel_loss!, normalize_columns!

export SparseSkipGramD, train, init, close_device

include("config.jl")
include("array_handler.jl")
include("samples.jl")
include("utils.jl")

type SparseSkipGramD{Tensor} # Tensor could be HostArray{Float32, 2}
    α::Tensor
    β::Tensor
    A::Tensor

    # the following fields are training parameters.
    m::Int
    k::Int
    lr::Float32
    lr0::Float32
    λ::Float32
    ɛ::Float32

    # maps
    idx2word::Array{String, 1}
    word2idx::Dict{String, UInt32}

    # the following fields are named memory for training
    ∇αβ::Tensor
    ᾱ::Tensor
    β̄::Tensor
    a⃗::Tensor
    b⃗::Tensor
    γ::Tensor
    σ::Tensor
    ∇α::Tensor
    M::Tensor
    ∇β::Tensor
    ∇A::Tensor
    idxes::Tensor
    sign::Tensor

    function SparseSkipGramD(
        num_base::Integer,     # number of atom words
        num_vocab::Integer,    # number of vocabulary
        mini_batch::Integer,     # mini-batch size
        negative::Integer,        # number of negative samples per positive sample
        lr::AbstractFloat=0.01,       # learning rate
        λ::AbstractFloat=0.01,         # sparseness weight
        ɛ::AbstractFloat=0.01,
        init_model::String=""   # dumped init_model.
    )
        @assert num_base ≤ num_vocab "The number of atom words must be less than the vocabulary size"

        α_ = randn(Float32, num_base, num_vocab) ./ sqrt(num_base)

        i2w = Array{String, 1}()
        w2i = Dict{String, UInt32}()

        if init_model != ""
            e, c, i2w = jldopen(init_model, "r") do f
                (read(f, "embeddings"), read(f, "contexts"),
                 read(f, "idx2word"))
            end
            @assert num_vocab ≤ length(i2w) "Initial model must contains as many vocabularies as the model to be trained."

            d = size(e, 1)
            B_ = randn(Float32, num_base, num_base) ./ sqrt(num_base)
            C_ = randn(Float32, num_base, num_base) ./ sqrt(num_base)

            i2w = i2w[1:num_vocab]
            for (i, w) in enumerate(i2w)
                w2i[w] = i
            end

            α_[1:num_base, 1:num_base] = eye(num_base)
            copy!(B_, @view e[:, 1:num_base])
            copy!(C_, @view c[:, 1:num_base])
            A_ = B_' * C_
        else
            A_ = abs(randn(Float32, (num_base, num_base)) ./ sqrt(num_base))
        end

        α = Tensor(α_)
        β = α
        A = Tensor(A_)


        ∇αβ = Tensor(Float32, num_base*(negative+2)*mini_batch)
        ∇α, ∇β = splitarray(∇αβ, (num_base, mini_batch),
                                   (num_base, negative+1, mini_batch))

        ᾱ = Tensor(Float32, num_base, mini_batch)
        β̄ = Tensor(Float32, num_base, negative+1, mini_batch)
        a⃗ = Tensor(Float32, num_base, mini_batch)
        b⃗ = Tensor(Float32, num_base, negative+1, mini_batch)
        γ = Tensor(Float32, negative+1, mini_batch)
        σ = Tensor(Float32, negative+1, mini_batch)
        M = Tensor(Float32, num_base, mini_batch)
        ∇A = Tensor(Float32, num_base, num_base)

        idxes = Tensor(UInt32, (negative+2)*mini_batch)

        fill!(∇αβ, 0.0)
        fill!(ᾱ, 0.0)
        fill!(β̄, 0.0)
        fill!(a⃗, 0.0)
        fill!(b⃗, 0.0)
        fill!(γ, 0.0)
        fill!(σ, 0.0)
        fill!(M, 0.0)
        fill!(∇A, 0.0)
        fill!(idxes, 0)

        tmp = -ones(Float32, (negative+1, mini_batch))
        tmp[1, :] = 1.0f0
        s = Tensor(tmp)

        new{Tensor}(α, β, A,                          # model parameters
            mini_batch, negative, lr, lr, λ, ɛ,       # training parameters
            i2w, w2i,                                 # maps
            ∇αβ, ᾱ, β̄, a⃗, b⃗, γ, σ, ∇α, M, ∇β, ∇A, idxes, s
        )
    end
end

function save(model::SparseSkipGramD{Array}, fname::String; save_maps=false)
    jldopen(fname, "w") do f
        write(f, "α", model.α)
        write(f, "A", model.A)

        # the following fields are training parameters.
        write(f, "m", model.m)
        write(f, "k", model.k)
        write(f, "lr", model.lr0)
        write(f, "λ", model.λ)
        write(f, "ɛ", model.ɛ)

        # maps
        if save_maps == true
            write(f, "idx2word", model.idx2word)
            write(f, "word2idx", model.word2idx)
        end
    end
end

num_vocab(model::SparseSkipGramD) = size(model.α, 2)
num_base(model::SparseSkipGramD) = size(model.α, 1)
# num_vecdim(model::SparseSkipGramD) = size(model.B, 1)
num_minibatch(model::SparseSkipGramD) = size(model.a⃗, 2)
num_negative(model::SparseSkipGramD) = size(model.b⃗, 2) - 1

function to_device{M <: Union{SparseSkipGramD{Array},
                    SparseSkipGramD{HostArray}}}(model::M)
    num_base_ = num_base(model)
    num_vocab_ = num_vocab(model)
    mini_batch = num_minibatch(model)
    negative = num_negative(model)

    d_model = SparseSkipGramD{CudaArray}(num_base_, num_vocab_, mini_batch,
                                        negative, model.lr, model.λ, model.ɛ)
    for fn in fieldnames(typeof(d_model))
        if fn == :β
            continue
        end
        dest = getfield(d_model, fn)
        if isa(dest, CudaArray)
             src = getfield(model, fn)
             copy!(dest, src)
        end
    end
    return d_model
end

@generated function get_training_model(model::SparseSkipGramD)
    if BACKEND == "GPU"
        return :(to_device(model))
    else
        return :(model)
    end
end

function compute_grads_and_updateA!(model, batch)
    const m = model.m
    const k = model.k
    const A = model.A
    const M = model.M
    w, c = batch
    const nb = num_base(model)
    const ᾱ = takelast!(model.ᾱ, model.α, w)
    const β̄ = takelast!(model.β̄, model.β, c)
    const a⃗ = model.a⃗
    const b⃗ = model.b⃗
    const γ = model.γ
    const ɛ = model.ɛ
    const ∇α = model.∇α
    const ∇β = model.∇β
    const ∇A = model.∇A

    flat_b⃗ = reshape(b⃗, (nb, (k+1)*m))
    gemm!('T', 'N', 1.0f0, A, ᾱ, 0.0f0, a⃗)
    gemm!('N', 'N', 1.0f0, A, reshape(β̄, (nb, (k+1)*m)), 0.0f0, flat_b⃗)

    α_ = reshape(ᾱ, (nb, 1, m))
    γ_ = reshape(γ, (k+1, 1, m))
    gemm_strided_batched!('T', 'N', 1.0f0, b⃗, α_, 0.0f0, γ_)
    gamma_kernel_loss!(γ, model.σ, model.sign)

    #∇α
    ∇α_ = reshape(∇α, (nb, 1, m))
    gemm_strided_batched!('N', 'N', 1.0f0, b⃗, γ_, 0.0f0, ∇α_)

    # ∇β
    a⃗_ = reshape(a⃗, (nb, 1, m))
    gemm_strided_batched!('N', 'T', 1.0f0, a⃗_, γ_, 0.0f0, ∇β)

    # ∇A
    M_ = reshape(M, (nb, 1, m))
    gemm_strided_batched!('N', 'N', 1.0f0, β̄, γ_, 0.0f0, M_)
    # gemm!('N', 'T', 1.0f0, ᾱ, M, 0.0f0, ∇A)
    # axpy!(ɛ, vec(A), vec(∇A))

    # update A
    gemm!('N', 'T', -model.lr, ᾱ, M, 1.0f0-model.lr*model.ɛ, A)
    nothing
end

function fetch_grad_sparse!(grads, model)
    copy!(grads, model.∇αβ)
    return grads
end

function gpu_sparse_update!(arr, grad, idxes, lr, lambda)
    # make sure the transferring completes.
    CUDArt.synchronize(CUDArt.null_stream)
    ccall((:sparse_update, vec_lib), Void, (Ptr{Float32}, Ptr{Float32},
        Cint, Cint, Ptr{Cuint}, Float32, Float32, Ptr{Void}, Cint),
        arr.ptr, grad.ptr, size(grad, 1), size(grad, 2),
        idxes.ptr, lr, lambda, null_stream.inner.handle, 256)
end

# function update_A!(A, gradA, lr, e, num_thread::Integer=256,
#     stream::Ptr{Void}=null_stream.inner.handle)
#
#     ccall((:update_A_neg, vec_lib), Void,
#           (Ptr{Cfloat}, Ptr{Cfloat}, Cfloat, Cfloat, Cint,
#            Ptr{Void}, Cint),
#            vec(A), vec(gradA), lr, e, length(A), stream, num_thread)
# end

@generated function proximal_update!(model, ∇αβ, idxes)
    if BACKEND == "GPU"
        quote
            const lr_::Float32 = model.lr
            dev = Int(DEVICE)
            siz_αβ = (num_base(model), length(idxes))
            d_∇αβ = CudaArray(model.∇αβ.ptr, siz_αβ, dev)
            copy!(d_∇αβ, ∇αβ)

            ptr_idxes = model.idxes.ptr
            d_idxes = CudaArray(ptr_idxes, (length(idxes),), dev)
            copy!(d_idxes, idxes)
            gpu_sparse_update!(model.α, d_∇αβ, d_idxes, lr_, model.λ)
        end
    else
        quote
            error("Not implemented")
            # lr_::Float32 = model.lr
            # cpu_sparse_update!(model.α, ∇αβ, idxes, lr_, model.λ)
        end
    end
end

function update_sparse_u1!(model, ∇αβ, grad_αβ, uniq_batch)
    idx, inv_idxes = uniq_batch

    ∇αβ = @view ∇αβ[:, 1:length(idx)]
    fill!(∇αβ, 0.0f0)

    # make sure the transferring completes.
    CUDArt.synchronize(CUDArt.null_stream)

    for (i, inv_idx) in enumerate(inv_idxes)
        ∇αβ[:, inv_idx] += grad_αβ[:, i]
    end

    proximal_update!(model, ∇αβ, idx)
    nothing
end

function update_sparse_u2!(model, ∇αβ, grad_αβ, uniq_batch)
    idx, inv_idxes = uniq_batch

    ∇αβ = @view ∇αβ[:, 1:length(idx)]
    fill!(∇αβ, 0.0f0)

    # make sure the transferring completes.
    CUDArt.synchronize(CUDArt.null_stream)

    for (i, inv_idx) in enumerate(inv_idxes)
        ∇αβ[:, inv_idx] += grad_αβ[:, i]
    end
    normalize_columns!(∇αβ)

    proximal_update!(model, ∇αβ, idx)
    nothing
end


@generated function sync_params!(cpu_model, model)
    if BACKEND == "CPU"
        return :(cpu_model)
    else
        quote
            for fn in (:α, :A)  #fieldnames(typeof(model))
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

function train(model; min_lr=1f-6, normalize_sp=false,
               saveratio=0.05, part=1.0, save_basename="D")

    global sampleFactory
    cpu_model = model
    model = to_device(cpu_model)
    beta = model.lr0 / min_lr - 1.0f0

    info(@sprintf("Train with lr=%g, normalize_sp=%s. Save after every %.2f%% of training. Use %.2f%% of the total training data. Save the trained model to \"%s*.jld\"", model.lr, string(normalize_sp), saveratio*100, part*100, save_basename))

    grads = HostArray(Float32, (num_base(model), (model.k+2)*model.m))
    grad_cache = HostArray(Float32, (num_base(model), (model.k+2)*model.m))

    factory = get(sampleFactory)
    const producer = factory.producer
    const progress_ = factory -> progress(factory)/part
    batch_set = Task(() -> producer(factory, model.m, model.k))

    const update_sparse! = normalize_sp ? update_sparse_u2! : update_sparse_u1!

    st = time()
    init_logger(model, st)

    old_batch = consume(batch_set)
    next_save_point = saveratio
    compute_grads_and_updateA!(model, old_batch)
    for batch in batch_set
        uniq_batch = invertable_unique(old_batch[1], old_batch[2])
        # synchronize(CUDArt.null_stream)
        # TODO: maybe another stream to handle the data transferring
        fetch_grad_sparse!(grads, model)
        update_sparse!(model, grad_cache, grads, uniq_batch)
        old_batch = batch
        compute_grads_and_updateA!(model, old_batch)
        prog = progress_(factory)
        if prog >= next_save_point
            if prog >= 1.0
                break
            end
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
end


end # end of module
