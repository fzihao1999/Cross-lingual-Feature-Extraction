output=~/data/wiki_data/ja-en-init/ja-en-k-com-results

for k in {1..9}
do
    echo "$k:" >> $output
    for i in {50..300..50}
    do
        echo $i
        export CUDA_VISIBLE_DEVICES="1"
        python map_emb_init_com.py --unsupervised ../data/wiki_data/vec/ja_3_10.vec ../data/wiki_data/vec/en.vec ../data/wiki_data/ja-en-init/ja.mapped.vec ../data/wiki_data/ja-en-init/en.mapped.vec --src_semvec ../data/wiki_data/sem_norm_vec/ja_sem_3_10.vec --trg_semvec ../data/wiki_data/sem_norm_vec/en_sem.vec --sem_dim $i --add_mat $k --validation ../data/wiki_data/crosslingual/dictionaries/ja-en.5000-6500.txt --verbos --log ../data/wiki_data/ja-en-init/ja-en-init.log --cuda
        eval=$(python eval_translation.py ../data/wiki_data/ja-en-init/ja.mapped.vec ../data/wiki_data/ja-en-init/en.mapped.vec -d ../data/wiki_data/crosslingual/dictionaries/ja-en.5000-6500.txt --retrieval csls --cuda)
        echo "$i:" >> $output
        echo $eval >> $output
    done
done