output=~/data/wiki_data/en-vi-base/en-vi_en1-10.txt

for i in {10..1}
do
    echo "$i"
    python ~/data/wiki_data/txt/divide.py --language en --number $i
    ~/fastText/fasttext skipgram -input ../data/wiki_data/txt/en.all -output ../data/wiki_data/vec_weak/en_$i -dim 300
    export CUDA_VISIBLE_DEVICES="0"
    python map_embeddings.py --unsupervised ../data/wiki_data/vec_weak/en_$i.vec ../data/wiki_data/vec/vi.vec ../data/wiki_data/en-vi-base/en.mapped.vec ../data/wiki_data/en-vi-base/vi.mapped.vec --validation ../data/wiki_data/crosslingual/dictionaries/en-vi.5000-6500.txt --verbose --log ../data/wiki_data/en-vi-base/en-vi-base.log --cuda
    eval=$(python eval_translation.py ../data/wiki_data/en-vi-base/en.mapped.vec ../data/wiki_data/en-vi-base/vi.mapped.vec -d ../data/wiki_data/crosslingual/dictionaries/en-vi.5000-6500.txt --retrieval csls --cuda)
    echo "$i:" >> $output
    echo $eval >> $output
    rm -rf ../data/wiki_data/vec_weak/en_$i.vec
    rm -rf ../data/wiki_data/vec_weak/en_$i.bin
done