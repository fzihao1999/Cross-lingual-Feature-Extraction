# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from cupy_utils import *

import numpy as np


def read(file, threshold=0, vocabulary=None, dtype='float'):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
        elif word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
    return (words, matrix, dim) if vocabulary is None else (words, np.array(matrix, dtype=dtype))

def combine(pre_words, sem_words, x_pre, x_sem, src_word2ind_sem, dim, dtype='float', sem_dim=300):
    sem_number = len(sem_words)
    pre_number = len(pre_words)
    words = []
    #matrix = np.empty((len(pre_words), dim+sem_dim), dtype=dtype)
    matrix = np.empty((len(pre_words), sem_dim), dtype=dtype)
    flag = {}
    j = 0
    for i in range(pre_number):
        word = pre_words[i]
        words.append(word)
        mx1 = x_pre[i]
        if word in src_word2ind_sem.keys():
            mx2 = x_sem[src_word2ind_sem[word]][:sem_dim]
        else:
            mx2 = np.zeros(sem_dim)
        #matrix[i] = np.hstack((mx1,mx2))
        matrix[i] = mx2
    '''for i in range(sem_number):
        word = sem_words[i]
        flag[word] = True
        mx1 = x_sem[i]
        mx2 = x_pre[src_word2ind_pre[word]]
        matrix[i] = np.hstack((mx1,mx2)) 
        words.append(word)
        j = i
        
    for i in range(pre_number):
        word = pre_words[i]
        if word in flag.keys():
            continue
        j += 1
        words.append(word)
        mx1 = x_pre[i]
        mx2 = np.zeros(dim)
        matrix[j] = np.hstack((mx2,mx1))'''
    return words, matrix      

def write(words, matrix, file):
    m = asnumpy(matrix)
    print('%d %d' % m.shape, file=file)
    for i in range(len(words)):
        print(words[i] + ' ' + ' '.join(['%.6g' % x for x in m[i]]), file=file)


def length_normalize(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, xp.newaxis]


def mean_center(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=0)
    matrix -= avg


def length_normalize_dimensionwise(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix**2, axis=0))
    norms[norms == 0] = 1
    matrix /= norms


def mean_center_embeddingwise(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=1)
    matrix -= avg[:, xp.newaxis]

def min_max_norm(matrix):
    xp = get_array_module(matrix)
    count = len(matrix)
    norms = (xp.max(matrix, axis=1) - xp.min(matrix, axis=1)).reshape(count, 1)
    norms[norms == 0] = 1
    return (matrix - xp.mean(matrix, axis=1).reshape(count, 1)) / norms

def min_norm(matrix):
    xp = get_array_module(matrix)
    count = len(matrix)
    norms = (xp.max(matrix, axis=1) - xp.min(matrix, axis=1)).reshape(count, 1)
    norms[norms == 0] = 1
    return (matrix - xp.min(matrix, axis=1).reshape(count, 1)) / norms

def normalize(matrix, actions):
    for action in actions:
        if action == 'unit':
            length_normalize(matrix)
        elif action == 'center':
            mean_center(matrix)
        elif action == 'unitdim':
            length_normalize_dimensionwise(matrix)
        elif action == 'centeremb':
            mean_center_embeddingwise(matrix)
