const _BACKEND = "GPU"
const _DEVICE = 0

const _device_pat = r"([A-Za-z]+)(\d*)"
if haskey(ENV, "DEVICE")
    dev = uppercase(ENV["DEVICE"])
elseif haskey(ENV, "device")
    dev = uppercase(ENV["device"])
else
    dev = @sprintf("%s%d", _BACKEND, _DEVICE)
end

m = match(_device_pat, dev)
const BACKEND = m.captures[1]
const DEVICE = length(m.captures[2])>0? parse(UInt8, m.captures[2]) : UInt8(0)
# const BACKEND = Ref{String}(_BACKEND)
# const DEVICE = Ref{UInt8}(_DEVICE)
m = nothing

if BACKEND == "GPU"
    import CUDArt: CudaArray
    typealias TrainArray CudaArray
    const vec_lib = joinpath(homedir(), "lib", "VecLib.so")
else
    typealias TrainArray Array
end

# warn(@sprintf("using %s-%d\nIf you want to change the device, please do it before loading %s/utils.jl with function `change_backend(dev::String)`", _BACKEND, _DEVICE, dirname(@__FILE__)))
#
# function change_backend(dev)
#     global BACKEND
#     global DEVICE
#     m = match(_device_pat, dev)
#     backend = m.captures[1]
#     nb = length(m.captures[2])>0? parse(UInt8, m.captures[2]) : UInt8(0)
#     BACKEND[] = backend
#     DEVICE[] = nb
#     return (backend, nb)
# end
