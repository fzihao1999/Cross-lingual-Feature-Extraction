output=~/data/wiki_data/en-ja-base/en-ja_ja1-10.txt

for i in {10..1}
do
    echo "$i"
    python ~/data/wiki_data/txt/divide_ja.py --language ja --number $i
    ~/fastText/fasttext skipgram -input ../data/wiki_data/txt/ja.all -output ../data/wiki_data/vec_weak/ja_$i -dim 300
    export CUDA_VISIBLE_DEVICES="2"
    python map_embeddings.py --unsupervised ../data/wiki_data/vec/en.vec ../data/wiki_data/vec_weak/ja_$i.vec ../data/wiki_data/en-ja-base/en.mapped.vec ../data/wiki_data/en-ja-base/ja.mapped.vec --validation ../data/wiki_data/crosslingual/dictionaries/en-ja.5000-6500.txt --verbose --log ../data/wiki_data/en-ja-base/en-ja-base.log --cuda
    eval=$(python eval_translation.py ../data/wiki_data/en-ja-base/en.mapped.vec ../data/wiki_data/en-ja-base/ja.mapped.vec -d ../data/wiki_data/crosslingual/dictionaries/en-ja.5000-6500.txt --retrieval csls --cuda)
    echo "$i:" >> $output
    echo $eval >> $output
    rm -rf ../data/wiki_data/vec_weak/ja_$i.vec
    rm -rf ../data/wiki_data/vec_weak/ja_$i.bin
done