import numpy as np
import time
import json

def eval_fre(path):
    dict = {}
    file = open(path,'r',encoding='utf8')
    for line in file.readlines():
        line = line.rstrip()
        words = line.split(" ")
        for word in words:
            if word in dict.keys():
                dict[word] += 1
            else:
                dict[word] = 1
    return sorted(dict.items(), key=lambda d: d[1], reverse=True)

def read_paneldata(path):
    en = []
    es = []
    file = open(path,'r',encoding='utf8')
    for line in file.readlines():
        line = line.rstrip()
        words = line.split(" ")
        en.append(words[0])
        es.append(words[1])
    return en, es

def eval_pos(path, test_word, std_word):
    test_dict = {}
    std_dict = {}
    file = open(path,'r',encoding='utf8')
    line_number = 0
    line_len = []
    time1 = time.time()
    for line in file.readlines():
        if line_number == 1000000:
            break
        pos = 0
        line = line.rstrip()
        words = line.split(" ")
        for word in words:
            if word in test_word:
                if word not in test_dict.keys():
                    test_dict[word] = []
                test_dict[word].append(str(line_number) + " " + str(pos))

            if word in std_word:
                if word not in std_dict.keys():
                    std_dict[word] = []
                std_dict[word].append(str(line_number) + " " + str(pos))
            pos += len(word) + 1
        line_number += 1
        line_len.append(pos)
    return test_dict, std_dict, line_len

def get_pos(pos):
    pos = pos.split(" ")
    return int(pos[0]), int(pos[1])

def check(test_line, test_pos, std_line, std_pos):
    if test_line < std_line:
        return True
    if test_line == std_line:
        if test_pos <= std_pos:
            return True
        else:
            return False
    if test_line > std_line:
        return False

def calc_dis(test_line, test_pos, std_line, std_pos, linelen, word_len):
    dis = 0
    first_line = True
    while test_line < std_line:
        if first_line == True:
            dis += linelen[test_line] - test_pos + word_len
            print(linelen[test_line], test_pos, word_len, dis)
            first_line = False
            test_line += 1
        else:
            dis += linelen[test_line]
            test_line += 1
    if first_line == True:
        dis += std_pos - test_pos + word_len
    else:
        dis += std_pos
    return dis

def calc_Lij(test, std, linelen, word_len):
    i = 0
    j = 0
    len1 = len(test)
    len2 = len(std)
    n = 0
    L = 0
    while True:
        test_line, test_pos = get_pos(test[i])
        std_line, std_pos = get_pos(std[j])
        while check(test_line, test_pos, std_line, std_pos) == False:
            j += 1
            if j >= len2:
                break 
        o = 1
        while check(test_line, test_pos, std_line, std_pos) == True:
            i += 1
            if i >= len1:
                o = 0
                break
        i -= o

        #print(test_line, test_pos, std_line, std_pos, calc_dis(test_line, test_pos, std_line, std_pos, linelen, word_len))
        if check(test_line, test_pos, std_line, std_pos) == True:
            #print(test_line, test_pos, std_line, std_pos, calc_dis(test_line, test_pos, std_line, std_pos, linelen, word_len))
            n += 1
            L += calc_dis(test_line, test_pos, std_line, std_pos, linelen, word_len)
        
        i += 1
        j += 1
        if i >= len1 or j >= len2:
            break
    return float(n)/(float(L)+1)

def calc_L(p1500, p5000, dict_test, dict_std, linelen):
    #test = list(dict_test.keys())
    #std = list(dict_std.keys())
    #Lij = np.zeros((len(p1500),len(p5000)))
    Lij = np.zeros((1500,5000))
    print(len(p1500), len(p5000))
    for i in range(1500):
        for j in range(5000):
            if p1500[i] in dict_test.keys() and p5000[j] in dict_std.keys():
                Lij[i,j] = calc_Lij(dict_test[p1500[i]], dict_std[p5000[j]], linelen, len(p1500[i]))
            else:
                Lij[i,j] = 0
            print(i,j,Lij[i,j])
    return Lij, p1500

def predict(Lij_en, en_words, Lij_es, es_words):
    dict_pre = {}
    Lij = np.dot(Lij_en, Lij_es.T).tolist()
    for i in range(len(Lij)):
        max_num = 0
        max_j = -1
        for j in range(len(Lij[i])):
            if max_j == -1:
                max_num = Lij[i][j]
                max_j = j
            else:
                if max_num < Lij[i][j]:
                    max_num = Lij[i][j]
                    max_j = j
        dict_pre[en_words[i]] = es_words[max_j]
    return dict_pre

def eval(dict_pre, path):
    file = open(path, 'r', encoding='utf8')
    correct = 0
    amount = 0
    for line in file.readlines():
        amount += 1
        line = line.rstrip()
        words = line.split(" ")
        if words[0] in dict_pre.keys():
            if dict_pre[words[0]] == words[1]:
                correct += 1
        print(correct, amount)
        if amount == 1500:
            break
    print(str(correct) + "/" + str(amount))
                
if __name__ == "__main__":
    #exit(0)
    #en_fre = eval_fre("./wiki_data/en/wiki.en.txt")[0:200000]
    #es_fre = eval_fre("./wiki_data/es/wiki.es.txt")[0:200000]
    pen, pes = read_paneldata("./crosslingual/dictionaries/en-es.5000-6500.txt")
    dict_en, dict_es = read_paneldata("./crosslingual/dictionaries/en-es.0-5000.txt")
    time1 = time.time()
    '''test_pos_en, std_pos_en, linelen_en = eval_pos("./wiki_data/en/wiki.en.txt", pen, dict_en)
    with open("test_pos_en.json", 'w', encoding='utf8') as file:
        file.write(json.dumps(test_pos_en))
    with open("std_pos_en.json", 'w', encoding='utf8') as file:
        file.write(json.dumps(std_pos_en))
    with open("linelen_en.txt", 'w', encoding='utf8') as file:
        for lens in linelen_en:
            file.write(str(lens) + '\n')
    time2 = time.time()
    print(time2 - time1)

    test_pos_es, std_pos_es, linelen_es = eval_pos("./wiki_data/es/wiki.es.txt", pes, dict_es)
    with open("test_pos_es.json", 'w', encoding='utf8') as file:
        file.write(json.dumps(test_pos_es))
    with open("std_pos_es.json", 'w', encoding='utf8') as file:
        file.write(json.dumps(std_pos_es))
    with open("linelen_es.txt", 'w', encoding='utf8') as file:
        for lens in linelen_es:
            file.write(str(lens) + '\n')
    time3 = time.time()
    print(time3 - time2)'''

    test_pos_en = {}
    with open("./wiki_data/test_pos_en.json", 'r', encoding='utf8') as file:
        test_pos_en = json.load(file)
    std_pos_en = {}
    with open("./wiki_data/std_pos_en.json", 'r', encoding='utf8') as file:
        std_pos_en = json.load(file)
    linelen_en = []
    with open("linelen_en.txt", 'r', encoding='utf8') as file:
        for line in file.readlines():
            linelen_en.append(int(line.rstrip()))
    test_pos_es = {}
    with open("./wiki_data/test_pos_es.json", 'r', encoding='utf8') as file:
        test_pos_es = json.load(file)
    std_pos_es = {}
    with open("./wiki_data/std_pos_es.json", 'r', encoding='utf8') as file:
        std_pos_es = json.load(file)
    linelen_es = []
    with open("linelen_es.txt", 'r', encoding='utf8') as file:
        for line in file.readlines():
            linelen_es.append(int(line.rstrip()))

    Lij_en, en_word = calc_L(pen, dict_en, test_pos_en, std_pos_en, linelen_en)
    Lij_es, es_word = calc_L(pes, dict_es, test_pos_es, std_pos_es, linelen_es)
    dict_pre = predict(Lij_en, en_word, Lij_es, es_word)
    eval(dict_pre, "./crosslingual/dictionaries/en-es.5000-6500.txt")
    time4 = time.time()
    print(time4 - time1)
