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

m = nothing

if BACKEND == "GPU"
    using CUDArt: CudaArray, device
    typealias TrainArray CudaArray
    device(DEVICE)
    const vec_lib = joinpath(homedir(), "lib", "VecLib.so")
else
    typealias TrainArray Array
end
