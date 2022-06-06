output=~/data/wiki_data/en-th-base/en-th_th1-10.txt

for i in {10..1}
do
    echo "$i"
    python ~/data/wiki_data/txt/divide_th.py --language th --number $i
    ~/fastText/fasttext skipgram -input ../data/wiki_data/txt/th.all -output ../data/wiki_data/vec_weak/th_$i -dim 300
    export CUDA_VISIBLE_DEVICES="1"
    python map_embeddings.py --unsupervised ../data/wiki_data/vec/en.vec ../data/wiki_data/vec_weak/th_$i.vec ../data/wiki_data/en-th-base/en.mapped.vec ../data/wiki_data/en-th-base/th.mapped.vec --validation ../data/wiki_data/crosslingual/dictionaries/en-th.5000-6500.txt --verbose --log ../data/wiki_data/en-th-base/en-th-base.log --cuda
    eval=$(python eval_translation.py ../data/wiki_data/en-th-base/en.mapped.vec ../data/wiki_data/en-th-base/th.mapped.vec -d ../data/wiki_data/crosslingual/dictionaries/en-th.5000-6500.txt --retrieval csls --cuda)
    echo "$i:" >> $output
    echo $eval >> $output
    rm -rf ../data/wiki_data/vec_weak/th_$i.vec
    rm -rf ../data/wiki_data/vec_weak/th_$i.bin
done