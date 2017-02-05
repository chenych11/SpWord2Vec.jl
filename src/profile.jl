using SpWord2Vec

init = SpWord2Vec.SpSkipGramA.init
sampleFactory = SpWord2Vec.SpSkipGramA.sampleFactory

init("/home/cyc/Data/project_data/SpSkipGram/pairs-wiki-1based-20k.idx")
factory = get(sampleFactory)

const producer = factory.producer
batch_set = Task(() -> producer(factory, 1024, 15))

st = time()
for i=1:10000
    consume(batch_set)
end
en = time()

speed = factory.consumed[]/(en-st)
println("speed: $speed bytes/s")

init("/home/cyc/Data/project_data/SpSkipGram/wiki-sg-norm-lc-drop.bz2")

factory = get(sampleFactory)

const producer_ = factory.producer
batch_set = Task(() -> producer_(factory, 1024, 15))

st = time()
for i=1:1000
    consume(batch_set)
end
en = time()

speed = factory.consumed[]/(en-st)
println("speed: $speed lines/s")


init("/home/cyc/Data/project_data/SpSkipGram/wiki-sg-norm-lc-drop.bz2")
factory = get(sampleFactory)
const producer__ = factory.producer
batch_set = Task(() -> producer__(factory, 1024, 7))

st = time()
for i=1:1000
    consume(batch_set)
end
en = time()

speed = factory.consumed[]/(en-st)
println("speed: $speed lines/s")
