using CycUtils: multinomial
using Iterators: chain, partition
using JLD: jldopen

type SampleFactory
    material :: String
    amount :: UInt64
    consumed :: Ref{UInt64}
    neg_prob :: Vector{Float32}
    producer :: Function
    function SampleFactory(material::String, neg_prob::Vector{Float32})
        global batch_producer
        global line_batch_producer
        if endswith(material, ".idx")
            amt = open(material) do f
                seekend(f)
                position(f)
            end
            return new(material, amt, Ref{UInt64}(0), neg_prob, batch_producer)
        end

        if !endswith(material, ".bz2")
            error("only accept bzip2 compressed plain text as corpus")
        end
        dn = dirname(material)
        fn = basename(material)[1:end-4]
        mn = joinpath(dn, fn * "-meta.jld")
        if isfile(mn)
            lines = jldopen(mn) do f
                read(f, "lines")
            end
        else
            nl, nw, nb = split(readstring(pipeline(`bzcat $material`, `wc`)))
            nl = parse(Int, nl)
            nw = parse(Int, nw)
            nb = parse(Int, nb)
            lines = nl
            jldopen(mn, "w") do f
                write(f, "lines", lines)
                write(f, "words", nw)
                write(f, "bytes", nb)
            end
        end
        return new(material, lines, Ref{UInt64}(0), neg_prob,
                   line_batch_producer)
    end
end

progress(factory::SampleFactory) = factory.consumed[]/factory.amount

const sampleFactory = Ref{Nullable{SampleFactory}}(Nullable{SampleFactory}())
const genBufFac = Ref{UInt32}(0xa)

function line_sampler(factory, m, k, word2idx, window=6)
    wsize = 1:window
    fn = factory.material
    lines = eachline(`bzcat $fn`)
    for line in lines
        sent = [word2idx[w] for w in split(line) if haskey(word2idx, w)]
        lsent = length(sent)
        for i = 1:lsent
            w = rand(wsize)
            st = max(1, i-w)
            en = min(lsent, i+w)

            for j in chain(st:i-1, i+1:en)
                produce((sent[i], sent[j]))
            end
        end
        factory.consumed[] += 1
    end
end

function line_batch_producer(factory, m, k)
    global init_model
    global genBufFac
    const multiplier = get(genBufFac)
    draw! = multinomial(factory.neg_prob)
    const nb_samples = m * multiplier
    idxes = collect(1:nb_samples)

    bigbatch_w = zeros(UInt32, nb_samples)
    bigbatch_c = zeros(UInt32, k+1, nb_samples)

    word2idx = jldopen(init_model) do f
        read(f, "word2idx")
    end
    corpus = Task(()->line_sampler(factory, m, k, word2idx))
    for x in partition(corpus, nb_samples)
        for (i, wc) in enumerate(x)
            w, c = wc
            bigbatch_w[i] = w
            bigbatch_c[1, i] = c
        end
        draw!(@view bigbatch_c[2:end, :])
        shuffle!(idxes)
        bigbatch_w = bigbatch_w[idxes]
        bigbatch_c = bigbatch_c[:, idxes]
        for st in 1:m:nb_samples
            en = st + m - 1
            produce((bigbatch_w[st:en], bigbatch_c[:, st:en]))
        end
    end
end

function batch_producer(factory, m, k)
    global genBufFac
    const multiplier = get(genBufFac)
    draw! = multinomial(factory.neg_prob)
    const nb_samples = m * multiplier
    idxes = collect(1:(2*nb_samples))
    buf = zeros(UInt32, (2, length(idxes)))
    const nbblk = sizeof(UInt32) * 2 * nb_samples

    open(factory.material) do f
        nbtotoal = factory.amount

        for i = 1:div(nbtotoal, nbblk)
            buf_pos = read(f, UInt32, (2, nb_samples))
            factory.consumed[] += nbblk
            buf[:, 1:nb_samples] = buf_pos
            buf[1, nb_samples+1:end] = buf_pos[2, :]
            buf[2, nb_samples+1:end] = buf_pos[1, :]
            shuffle!(idxes)
            buf_o = buf
            buf = reshape(buf_o[:, idxes], (2, m, 2*multiplier))
            for kk = 1:2*multiplier
                c = zeros(UInt32, (k+1, m))
                w = buf[1, :, kk]
                c[1, :] = buf[2, :, kk]
                draw!(@view c[2:end, :])
                produce((w, c))
            end
            buf = buf_o
        end
    end
    nothing
end
