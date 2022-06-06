output=~/data/wiki_data/en-zh-base/en-zh_zh1-10.txt

for i in {10..1}
do
    echo "$i"
    python ~/data/wiki_data/txt/divide_zh.py --language zh --number $i
    ~/fastText/fasttext skipgram -input ../data/wiki_data/txt/zh.all -output ../data/wiki_data/vec_weak/zh_$i -dim 300
    export CUDA_VISIBLE_DEVICES="2"
    python map_embeddings.py --unsupervised ../data/wiki_data/vec/en.vec ../data/wiki_data/vec_weak/zh_$i.vec ../data/wiki_data/en-zh-base/en.mapped.vec ../data/wiki_data/en-zh-base/zh.mapped.vec --validation ../data/wiki_data/crosslingual/dictionaries/en-zh.5000-6500.txt --verbose --log ../data/wiki_data/en-zh-base/en-zh-base.log --cuda
    eval=$(python eval_translation.py ../data/wiki_data/en-zh-base/en.mapped.vec ../data/wiki_data/en-zh-base/zh.mapped.vec -d ../data/wiki_data/crosslingual/dictionaries/en-zh.5000-6500.txt --retrieval csls --cuda)
    echo "$i:" >> $output
    echo $eval >> $output
    rm -rf ../data/wiki_data/vec_weak/zh_$i.vec
    rm -rf ../data/wiki_data/vec_weak/zh_$i.bin
done