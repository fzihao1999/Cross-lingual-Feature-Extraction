output=~/data/wiki_data/vi-en-init/vi-en-nonorm-results

for k in {0..10}
do
    echo "$k:" >> $output
    for i in {50..300..50}
    do
        echo $i
        export CUDA_VISIBLE_DEVICES="2"
        python map_emb_init_com.py --unsupervised ../data/wiki_data/vec/vi.vec ../data/wiki_data/vec/en.vec ../data/wiki_data/vi-en-init/vi.mapped.vec ../data/wiki_data/vi-en-init/en.mapped.vec --src_semvec ../data/wiki_data/sem_norm_vec/vi_sem_2000.vec --trg_semvec ../data/wiki_data/sem_norm_vec/en_sem.vec --sem_dim $i --add_mat $k --validation ../data/wiki_data/crosslingual/dictionaries/vi-en.5000-6500.txt --verbos --log ../data/wiki_data/vi-en-init/vi-en.log --cuda
        eval=$(python eval_translation.py ../data/wiki_data/vi-en-init/vi.mapped.vec ../data/wiki_data/vi-en-init/en.mapped.vec -d ../data/wiki_data/crosslingual/dictionaries/vi-en.5000-6500.txt --retrieval csls --cuda)
        echo "$i:" >> $output
        echo $eval >> $output
    done
done