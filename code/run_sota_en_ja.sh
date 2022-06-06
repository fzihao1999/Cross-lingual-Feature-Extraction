output=~/data/wiki_data/en-ja-sota/en-ja-results

for k in {50..300..50}
do
    python pca.py -i ../data/wiki_data/vec/ja_3_10.vec -o ../data/wiki_data/pca_vec/ja.vec --end $k
    python pca.py -i ../data/wiki_data/vec/en.vec -o ../data/wiki_data/pca_vec/en.vec --end $k
    export CUDA_VISIBLE_DEVICES="0"
    if [ $k -eq '50' ];then
        #echo "1"
        #echo $k
        python map_emb_sota.py --unsupervised ../data/wiki_data/pca_vec/en.vec ../data/wiki_data/pca_vec/ja.vec ../data/wiki_data/en-ja-sota/en.mapped.vec ../data/wiki_data/en-ja-sota/ja.mapped.vec --validation ../data/wiki_data/crosslingual/dictionaries/en-ja.5000-6500.txt --verbose --log ../data/wiki_data/en-ja-sota/en-ja-sota.log --cuda
        python eval_sota.py ../data/wiki_data/en-ja-sota/en.mapped.vec ../data/wiki_data/en-ja-sota/ja.mapped.vec --retrieval csls --output ../data/wiki_data/pca_vec/en-ja-dic.txt --cuda
        eval=$(python eval_translation.py ../data/wiki_data/en-ja-sota/en.mapped.vec ../data/wiki_data/en-ja-sota/ja.mapped.vec -d ../data/wiki_data/crosslingual/dictionaries/en-ja.5000-6500.txt --retrieval csls --cuda)
        echo "$k:" >> $output
        echo $eval >> $output
    else
        if [ $k -eq '150' ];then
            continue
        fi
        if [ $k -eq '250' ];then
            continue
        fi
        python map_emb_sota.py --semi_supervised ../data/wiki_data/pca_vec/en-ja-dic.txt ../data/wiki_data/pca_vec/en.vec ../data/wiki_data/pca_vec/ja.vec ../data/wiki_data/en-ja-sota/en.mapped.vec ../data/wiki_data/en-ja-sota/ja.mapped.vec --validation ../data/wiki_data/crosslingual/dictionaries/en-ja.5000-6500.txt --verbose --log ../data/wiki_data/en-ja-sota/en-ja.log --cuda
        python eval_sota.py ../data/wiki_data/en-ja-sota/en.mapped.vec ../data/wiki_data/en-ja-sota/ja.mapped.vec --retrieval csls --output ../data/wiki_data/pca_vec/en-ja-dic.txt --cuda
        eval=$(python eval_translation.py ../data/wiki_data/en-ja-sota/en.mapped.vec ../data/wiki_data/en-ja-sota/ja.mapped.vec -d ../data/wiki_data/crosslingual/dictionaries/en-ja.5000-6500.txt --retrieval csls --cuda)
        echo "$k:" >> $output
        echo $eval >> $output
    fi
done