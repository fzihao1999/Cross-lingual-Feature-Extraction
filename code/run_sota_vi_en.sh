output=~/data/wiki_data/vi-en-sota/vi-en-results

for k in {50..300..50}
do
    python pca.py -i ../data/wiki_data/vec/vi.vec -o ../data/wiki_data/vi-en-sota/vi.vec --end $k
    python pca.py -i ../data/wiki_data/vec/en.vec -o ../data/wiki_data/vi-en-sota/en.vec --end $k
    export CUDA_VISIBLE_DEVICES="3"
    if [ $k -eq '50' ];then
        #echo "1"
        #echo $k
        python map_emb_sota.py --unsupervised ../data/wiki_data/vi-en-sota/vi.vec ../data/wiki_data/vi-en-sota/en.vec ../data/wiki_data/vi-en-sota/vi.mapped.vec ../data/wiki_data/vi-en-sota/en.mapped.vec --validation ../data/wiki_data/crosslingual/dictionaries/vi-en.5000-6500.txt --verbose --log ../data/wiki_data/vi-en-sota/vi-en-sota.log --cuda
        python eval_sota.py ../data/wiki_data/vi-en-sota/vi.mapped.vec ../data/wiki_data/vi-en-sota/en.mapped.vec --retrieval csls --output ../data/wiki_data/pca_vec/vi-en-dic.txt --cuda
        eval=$(python eval_translation.py ../data/wiki_data/vi-en-sota/vi.mapped.vec ../data/wiki_data/vi-en-sota/en.mapped.vec -d ../data/wiki_data/crosslingual/dictionaries/vi-en.5000-6500.txt --retrieval csls --cuda)
        echo "$k:" >> $output
        echo $eval >> $output
    else
        if [ $k -eq '150' ];then
            continue
        fi
        if [ $k -eq '250' ];then
            continue
        fi
        python map_emb_sota.py --semi_supervised ../data/wiki_data/pca_vec/vi-en-dic.txt ../data/wiki_data/vi-en-sota/vi.vec ../data/wiki_data/vi-en-sota/en.vec ../data/wiki_data/vi-en-sota/vi.mapped.vec ../data/wiki_data/vi-en-sota/en.mapped.vec --validation ../data/wiki_data/crosslingual/dictionaries/vi-en.5000-6500.txt --verbose --log ../data/wiki_data/vi-en-sota/vi-en.log --cuda
        python eval_sota.py ../data/wiki_data/vi-en-sota/vi.mapped.vec ../data/wiki_data/vi-en-sota/en.mapped.vec --retrieval csls --output ../data/wiki_data/pca_vec/vi-en-dic.txt --cuda
        eval=$(python eval_translation.py ../data/wiki_data/vi-en-sota/vi.mapped.vec ../data/wiki_data/vi-en-sota/en.mapped.vec -d ../data/wiki_data/crosslingual/dictionaries/vi-en.5000-6500.txt --retrieval csls --cuda)
        echo "$k:" >> $output
        echo $eval >> $output
    fi
done