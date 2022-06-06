output=~/data/wiki_data/th-en-init/th-en-comonly-results

for k in {0..10..10}
do
    echo "$k:" >> $output
    for i in {50..300..50}
    do
        echo $i
        export CUDA_VISIBLE_DEVICES="3"
        python map_emb_init_com.py --unsupervised ../data/wiki_data/vec/th_1_10.vec ../data/wiki_data/vec/en.vec ../data/wiki_data/th-en-init/th.mapped.vec ../data/wiki_data/th-en-init/en.mapped.vec --src_semvec ../data/wiki_data/sem_norm_vec/th_sem_1_10.vec --trg_semvec ../data/wiki_data/sem_norm_vec/en_sem.vec --sem_dim $i --add_mat $k --validation ../data/wiki_data/crosslingual/dictionaries/th-en.5000-6500.txt --verbos --log ../data/wiki_data/th-en-init/th-en-$k.$i.log --cuda
        eval=$(python eval_translation.py ../data/wiki_data/th-en-init/th.mapped.vec ../data/wiki_data/th-en-init/en.mapped.vec -d ../data/wiki_data/crosslingual/dictionaries/th-en.5000-6500.txt --retrieval csls --cuda)
        echo "$i:" >> $output
        echo $eval >> $output
    done
done