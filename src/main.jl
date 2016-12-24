using SpWord2Vec
using ArgParse

function push_arg!(args, parsed_args, arg)
    if parsed_args[arg] != nothing
        push!(args, (Symbol(arg), parsed_args[arg]))
    end
end

function parse_commandline()
    s = ArgParseSettings(autofix_names=true)
    @add_arg_table s begin
        "--model", "-m"
            help = "Choose the model to train. Possible values are A, B, C, D and E"
            default = "A"
        "--corpus", "-c"
            help = "corpus file name"
            arg_type = String
            required = true
        "--lambda"
            help = "sparseness weight"
            arg_type = Float32
            default = 1.0f-2
        "--varepsilon"
            help = "L2 weight for matrix A in model [D]"
            arg_type = Float32
            default = 1.0f-2
        "--lr", "-a"
            help = "learning rate"
            arg_type = Float32
            default = 1.0f-3
        "--min-lr", "-b"
            help = "Minimum learning rate"
            arg_type = Float32
            default = 1.0f-4
        "--negative"
            help = "number negative samples per positive sample"
            arg_type = Int
            default = 15
        "--pretrain", "-p"
            help = "The ratio of data used to pre-train the sparse representations"
            arg_type = Float32
            default = 0.0f0
            dest_name = "preratio"
        "--every", "-e"
            help = "Dictionary parameters update frequencey"
            arg_type = Int
            default = 1
        "--normalize-sp"
            help = "normalize sparse parameter gradients or not"
            action = :store_true
            default = false
        "--normalize-dict"
            help = "normalize dictionary parameter gradients or not"
            action = :store_true
            default = false
        "--save-name"
            help = "save name"
            dest_name = "save_basename"
        "--mini-batch"
            help = "mini-batch size"
            arg_type = Int
            default = 1024
        "--saveratio"
            help = "save ratio"
            arg_type = Float32
            default = 0.05f0
        "--part"
            help = "training data ratio"
            arg_type = Float32
            default = 1.0f0
        "--vecdim", "-d"
            help = "words dimensions"
            arg_type = Int
            default = 200
        "--num-atom"
            help = "number of atoms"
            arg_type = Int
            default = 1024
        "--num-vocab"
            help = "number of vocabulary to keep"
            arg_type = Int
            default = 20_000
    end
    return parse_args(ARGS, s)
end

parsed_args = parse_commandline()
@eval SpSkipGram = SpWord2Vec.$(Symbol(:SpSkipGram, parsed_args["model"]))

train = SpSkipGram.train
init = SpSkipGram.init
close_device = SpSkipGram.close_device
@eval SparseSkipGram = SpSkipGram.$(Symbol(:SparseSkipGram, parsed_args["model"]))

corpus = parsed_args["corpus"]
d = parsed_args["vecdim"]
nbase = parsed_args["num_atom"]
nvocab = parsed_args["num_vocab"]
mini_batch = parsed_args["mini_batch"]
negative = parsed_args["negative"]
lr = parsed_args["lr"]
lambda = parsed_args["lambda"]
varepsilon = parsed_args["varepsilon"]

init(corpus)
args = Any[]

if parsed_args["model"] == "D"
    model = SparseSkipGram{Array}(
        nbase,              # number of atom words
        nvocab,             # number of vocabulary
        mini_batch,         # mini-batch size
        negative,           # number of negative samples per positive sample
        lr,                 # learning rate
        lambda,             # sparseness weight
        varepsilon,
        SpSkipGram.init_model)  # dumped init_model.
    for arg in String["min_lr", "normalize_sp", "saveratio", "part",
                      "save_basename"]
        push_arg!(args, parsed_args, arg)
    end
else
    model = SparseSkipGram{Array}(
        d,                  # vector dimension
        nbase,              # number of atom words
        nvocab,             # number of vocabulary
        mini_batch,         # mini-batch size
        negative,           # number of negative samples per positive sample
        lr,                 # learning rate
        lambda,             # sparseness weight
        SpSkipGram.init_model)  # dumped init_model.

    for arg in String["min_lr", "normalize_sp", "normalize_dict", "every",
                      "preratio", "saveratio", "part", "save_basename"]
        push_arg!(args, parsed_args, arg)
    end
end

train(model; args...)

close_device()
