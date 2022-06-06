import argparse
from re import L
from cupy_utils import *
#import numpy as np

def length_normalize(matrix):
    xp = get_cupy()
    norms = xp.sqrt(xp.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, xp.newaxis]
    return matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='init Lij')
    parser.add_argument('--src', help='the language to init')
    parser.add_argument('--trg', help='the language to init')
    args = parser.parse_args()
    src_lang = args.src
    trg_lang = args.trg

    file_pre_src = open("./wiki_data/vec/"+src_lang+"_3_10.vec", 'r', encoding='utf8')
    file_sem_src = open("./wiki_data/sem_norm_vec/"+src_lang+"_sem_3_10.vec", 'r', encoding='utf8')
    file_pre_trg = open("./wiki_data/vec/"+trg_lang+".vec", 'r', encoding='utf8')
    file_sem_trg = open("./wiki_data/sem_norm_vec/"+trg_lang+"_sem.vec", 'r', encoding='utf8')
    dict_file = open("./wiki_data/crosslingual/dictionaries/"+src_lang+"-"+trg_lang+".0-5000.txt", 'r', encoding='utf8')

    dic_fwd = {}
    dic_bwd = {}
    word2ind = {}
    number = 0
    for line in dict_file.readlines():
        words = line.split()
        if words[0] not in dic_fwd.keys():
            word2ind[words[0]] = number
            number += 1
            dic_fwd[words[0]] = words[1]
            dic_bwd[words[1]] = words[0]

    xp = get_cupy()

    flag = False
    src_pre_features = xp.empty((5000, 300), dtype='float32')
    for line in file_pre_src.readlines():
        if flag == False:
            flag = True
            continue
        word, vec = line.split(' ', 1)
        if word in dic_fwd.keys():
            src_pre_features[word2ind[word]] = xp.fromstring(vec, sep=' ', dtype='float32')

    flag = False
    src_sem_features = xp.empty((5000, 300), dtype='float32')
    for line in file_sem_src.readlines():
        if flag == False:
            flag = True
            continue
        word, vec = line.split(' ', 1)
        if word in dic_fwd.keys():
            src_sem_features[word2ind[word]] = xp.fromstring(vec, sep=' ', dtype='float32')

    flag = False
    trg_pre_features = xp.empty((5000, 300), dtype='float32')
    for line in file_pre_trg.readlines():
        if flag == False:
            flag = True
            continue
        word, vec = line.split(' ', 1)
        if word in dic_bwd.keys():
            trg_pre_features[word2ind[dic_bwd[word]]] = xp.fromstring(vec, sep=' ', dtype='float32')

    flag = False
    trg_sem_features = xp.empty((5000, 300), dtype='float32')
    for line in file_sem_trg.readlines():
        if flag == False:
            flag = True
            continue
        word, vec = line.split(' ', 1)
        if word in dic_bwd.keys():
            trg_sem_features[word2ind[dic_bwd[word]]] = xp.fromstring(vec, sep=' ', dtype='float32')

    src_pre_features = length_normalize(src_pre_features)
    src_sem_features = length_normalize(src_sem_features)
    trg_pre_features = length_normalize(trg_pre_features)
    trg_sem_features = length_normalize(trg_sem_features)

    u, s, vt = xp.linalg.svd(src_pre_features[:5000], full_matrices=False)
    xsim = (u*s).dot(u.T)
    u, s, vt = xp.linalg.svd(trg_pre_features[:5000], full_matrices=False)
    zsim = (u*s).dot(u.T)
    xsim.sort(axis=1)
    zsim.sort(axis=1)
    xsim = length_normalize(xsim)
    zsim = length_normalize(zsim)
    sim_pre = xsim.dot(zsim.T)

    u, s, vt = xp.linalg.svd(src_sem_features[:5000], full_matrices=False)
    xsim = (u*s).dot(u.T)
    u, s, vt = xp.linalg.svd(trg_sem_features[:5000], full_matrices=False)
    zsim = (u*s).dot(u.T)
    xsim.sort(axis=1)
    zsim.sort(axis=1)
    xsim = length_normalize(xsim)
    zsim = length_normalize(zsim)
    sim_sem = xsim.dot(zsim.T)
    sim_sem = src_sem_features.dot(trg_sem_features.T)

    sum_pre = 0.0
    sum_sem = 0.0

    for i in range(5000):
        sum_pre += sim_pre[i,i]
        sum_sem += sim_sem[i,i]
    
    print(sum_pre/5000.0, sum_sem/5000.0)

    
