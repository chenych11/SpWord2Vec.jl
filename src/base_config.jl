const data_root_dir = joinpath(homedir(), "Data", "project_data")
const train_data_full = joinpath(data_root_dir, "dep-wiki-1based")
const wordlist = joinpath(data_root_dir, "map-words")
const rellist = joinpath(data_root_dir, "map-dep-rel")
const wordmaps_full = joinpath(data_root_dir, "maps-word.jld")
const train_data_reduced = joinpath(data_root_dir,
                             "pairs-wiki-1based-global_index-30k")

const project_root = joinpath(data_root_dir, "SpSkipGram")
const train_data = joinpath(project_root, "pairs-wiki-1based-20k.idx")
const wordmaps = joinpath(project_root, "maps-word.jld")
const init_model = joinpath(project_root, "rw2vec-model-20k.jld")
