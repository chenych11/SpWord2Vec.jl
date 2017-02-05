#!/usr/bin/env bash
DATA_DIR=$HOME/Data/project_data/SpSkipGram
corpus="$DATA_DIR/wiki-sg-norm-lc-drop.bz2"

lr=0.001
minlr=0.0001
lmd=0.01
neg=7
prer=0.001
every=100
datap="0.4"
datapp=$(echo "scale=0; ${datap}*100"|bc)

echo 'export DEVICE="GPU0"'
for model in A B C E; do
    savename=\"$model.lr${lr}_${minlr}.λ${lmd}.neg${neg}.DLine${datapp}.nrm_sp.pre${prer}.U${every}\"
    echo "julia main.jl --model=$model --lr=$lr --min-lr=${minlr} --lambda=${lmd} --negative=${neg} --pretrain=${prer} --every=${every} --part=${datap} --normalize-sp --save-name=$savename  --corpus=$corpus 2>&1 | tee \"$DATA_DIR/log/${savename}.log\""
done

echo 'export DEVICE="GPU1"'
minlr=0.00005
for model in A B C E; do
    savename=\"$model.lr${lr}_${minlr}.λ${lmd}.neg${neg}.DLine${datapp}.nrm_sp.pre${prer}.U${every}\"
    echo "julia main.jl --model=$model --lr=$lr --min-lr=${minlr} --lambda=${lmd} --negative=${neg} --pretrain=${prer} --every=${every} --part=${datap} --normalize-sp --save-name=$savename  --corpus=$corpus 2>&1 | tee \"$DATA_DIR/log/${savename}.log\""
done


echo 'export DEVICE="GPU2"'
lr=0.005
minlr=0.0001
for model in A B; do
    savename=\"$model.lr${lr}_${minlr}.λ${lmd}.neg${neg}.DLine${datapp}.nrm_sp.pre${prer}.U${every}\"
    echo "julia main.jl --model=$model --lr=$lr --min-lr=${minlr} --lambda=${lmd} --negative=${neg} --pretrain=${prer} --every=${every} --part=${datap} --normalize-sp --save-name=$savename  --corpus=$corpus 2>&1 | tee \"$DATA_DIR/log/${savename}.log\""
done

lr=0.0025
minlr=0.00001
for model in A B; do
    savename=\"$model.lr${lr}_${minlr}.λ${lmd}.neg${neg}.DLine${datapp}.nrm_sp.pre${prer}.U${every}\"
    echo "julia main.jl --model=$model --lr=$lr --min-lr=${minlr} --lambda=${lmd} --negative=${neg} --pretrain=${prer} --every=${every} --part=${datap} --normalize-sp --save-name=$savename  --corpus=$corpus 2>&1 | tee \"$DATA_DIR/log/${savename}.log\""
done
