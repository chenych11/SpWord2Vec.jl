if BACKEND == "GPU"
    using CUDArt
    using CUBLAS
end
function init(material=train_data, num_buf_factor=20, prob_raise=0.75)
    global sampleFactory
    global genBufFac
    global init_model

    info(@sprintf("Using %s-%d", BACKEND, DEVICE))
    idx2wf = jldopen(init_model) do f
         read(f, "idx2wf")
    end

    for i = 1:length(idx2wf)
        @inbounds idx2wf[i] = idx2wf[i]^prob_raise
    end
    # sumx = sum(idx2wf)
    sumx = sum(idx2wf[i] for i = length(idx2wf):-1:1)
    for i = 1:length(idx2wf)
        @inbounds idx2wf[i] /= sumx
    end
    # info(length(idx2wf))

    info("Using $material as training data")
    set!(sampleFactory, SampleFactory(material, idx2wf))
    set!(genBufFac, convert(UInt32,num_buf_factor))
    if BACKEND == "GPU"
        CUDArt.device(DEVICE)
        # atexit(()->CUDArt.device_reset(DEVICE))
        CUDArt.init(DEVICE)
        # atexit(()->CUDArt.close(DEVICE))
        # atexit(CUBLAS.destroy)
    end
end

function close_device()
    if BACKEND == "GPU"
        CUBLAS.destroy()
        CUDArt.close(DEVICE)
        CUDArt.device_reset(DEVICE)
    end
end
