import numpy as np

def length_normalize(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, np.newaxis]
    return matrix

en_file = open("en_sem.vec", 'r', encoding='utf8')
en_lines = en_file.readlines()

zh_file = open("zh_sem_1_10.vec", 'r', encoding='utf8')
zh_lines = zh_file.readlines()

en_words = [',', 'one','years', 'school', 'history', 'think', 'american', 'culture']
zh_words = [',', '一', '年', '学校', '历史', '认为', '美国', '文化']
en = {}
zh = {}

for i in range(1, 100000):
    word, vec = en_lines[i].split(' ', 1)
    en[word] = vec
    enn = word

    word, vec = zh_lines[i].split(' ', 1)
    zh[word] = vec
    #print(enn, word)

dim = 100
en_feature = np.empty((8, dim), dtype='float32')
zh_feature = np.empty((8 ,dim), dtype='float32')

for i in range(8):
    #print(len(en[en_words[i]][0:dim]))
    #print(len(zh[zh_words[i]][0:dim]))
    #sexit()
    en_feature[i] = np.fromstring(en[en_words[i]], sep=' ', dtype='float32')[0:dim]
    zh_feature[i] = np.fromstring(zh[zh_words[i]], sep=' ', dtype='float32')[0:dim]

en_norm = length_normalize(en_feature)
zh_norm = length_normalize(zh_feature)

sim = en_norm.dot(zh_norm.T)

print(sim)


'''print("en")
for i in range(8):
    print(en_words[i] in en)

print("zh")
for i in range(8):
    print(zh_words[i] in zh)'''