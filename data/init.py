import time
import numpy as np
import json

def eval_zh():
    file_out = open("./zh/wiki_zh", 'w', encoding='utf8')
    for i in range(2):
        file_read = open("./zh/tokenized_wiki_0"+str(i), 'r', encoding='utf8')
        for line in file_read.readlines():
            if line.startswith(" < ") or line.startswith("< "):
                continue
            if len(line) <= 2:
                continue
            file_out.write(line)

def eval_en():
    file_out = open("./en/wiki_en", 'w', encoding='utf8')
    for i in range(16):
        if i < 10:
            file_name = "tokenized_wiki_0" + str(i)
        else:
            file_name = "tokenized_wiki_" + str(i)
        file_read = open("./en/"+file_name, 'r', encoding='utf8')
        for line in file_read.readlines():
            if line.startswith("&lt;"):
                continue
            if len(line) <= 2:
                continue
            file_out.write(line)

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
            #print(linelen[test_line], test_pos, word_len, dis)
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

def get_ij(pos_i, pos_j, word_len, linelen):
    i = 0
    j = 0
    len1 = min(1000, len(pos_i))
    len2 = min(1000, len(pos_j))
    #print(len1, len2)
    n = 0
    L = 0
    dis = []
    while True:
        i_line, i_pos = get_pos(pos_i[i])
        j_line, j_pos = get_pos(pos_j[j])
        #print(i, j, len1, len2)
        while check(i_line, i_pos, j_line, j_pos) == False:
            j += 1
            if j >= len2:
                break 
            j_line, j_pos = get_pos(pos_j[j])
        if j >= len2:
            break
        o = 1
        while check(i_line, i_pos, j_line, j_pos) == True:
            i += 1
            if i >= len1:
                o = 0
                break
            i_line, i_pos = get_pos(pos_i[i])
        i -= o
        if i >= len1:
            break
        #print(test_line, test_pos, std_line, std_pos, calc_dis(test_line, test_pos, std_line, std_pos, linelen, word_len))
        if check(i_line, i_pos, j_line, j_pos) == True:
            #print(test_line, test_pos, std_line, std_pos, calc_dis(test_line, test_pos, std_line, std_pos, linelen, word_len))
            dis.append(calc_dis(i_line, i_pos, j_line, j_pos, linelen, word_len))
            #n += 1
            #L += calc_dis(i_line, i_pos, j_line, j_pos, linelen, word_len)
        
        i += 1
        j += 1
        if i >= len1 or j >= len2:
            break
    return dis

def combine_dis(dis_fwd, dis_bwd):
    ans = 0
    number = 0
    dis = dis_fwd + dis_bwd
    dis.sort()
    for i in range(len(dis)):
        if dis[i] >= 1000:
            break
        number += 1
        ans += np.e**((float)(dis[i])*(-1)/50.0)
    if number == 0:
        return 0
    else:
        return ans/number

def find_top(word_pos):
    word_dict = {}
    for key in word_pos.keys():
        word_dict[key] = len(word_pos[key])
    word_sort = sorted(word_dict.items(), key=lambda item:item[1], reverse=True)
    word_list = []
    for i in range(500):
        word_list.append(word_sort[i][0])
    return word_list

def calc_vec(word_pos, lines):
    words = list(word_pos.keys())
    word_number = len(words)
    word_vec = {}
    topk = find_top(word_pos)
    print(topk[:10])
    for i in range(word_number):
        time1 = time.time()

        dis_i = []
        for j in range(500):
            #print(j)
            dis_fwd = get_ij(word_pos[words[i]], word_pos[topk[j]], len(words[i]), lines)
            dis_bwd = get_ij(word_pos[topk[j]], word_pos[words[i]], len(topk[j]), lines)
            dis_i.append(combine_dis(dis_fwd, dis_bwd))
        dis_i.sort(reverse=True)
        word_vec[words[i]] = dis_i[:300]

        time2 = time.time()
        print("word: ", words[i], "    time: ", str(time2-time1), "         ", str(i), "/", str(word_number))
        #exit()
    return word_vec

def eval_pos(path):
    word_dict = {}
    file = open(path,'r',encoding='utf8')
    line_number = 0
    line_len = []
    time1 = time.time()
    number = 0
    lines = file.readlines()
    print(len(lines))
    for line in lines:
        pos = 0
        line = line.rstrip()
        words = line.split(" ")
        for word in words:
            if word not in word_dict.keys():
                word_dict[word] = []
            word_dict[word].append(str(line_number) + " " + str(pos))
            pos += len(word) + 1
        line_number += 1
        line_len.append(pos)
        number += 1
        if number % 15000000 == 0:
            time2 = time.time()
            print(time2-time1)
            break
    return word_dict, line_len

def calc_Lij(word_dict, linelen):
    words = word_dict.keys()
    number = len(words)
    print(number)
    Lij = np.zeros((number,number))
    for i in range(number):
        time1 = time.time()
        for j in range(number):
            Lij[i,j] = get_ij(word_dict[words[i]], word_dict[words[j]], len(words[i]), linelen)
        time2 = time.time()
        print(time2-time1)

def init_Lij(load_file):
    time1 = time.time()
    if load_file == False:
        vi_word, vi_line = eval_pos("./wiki_data/txt/vi.all")
        with open("./wiki_data/pos/vi_pos.json", 'w', encoding='utf8') as file:
            file.write(json.dumps(vi_word))
        with open("./wiki_data/pos/vi.line", 'w', encoding='utf8') as file:
            for lens in vi_line:
                file.write(str(lens) + '\n') 
        time3 = time.time()
        print(time3 - time1)

        '''en_word, en_line = eval_pos("./en/wiki_en")
        with open("./wiki_data/pos/en_pos.json", 'w', encoding='utf8') as file:
            file.write(json.dumps(en_word))
        with open("./wiki_data/pos/en.line", 'w', encoding='utf8') as file:
            for lens in en_line:
                file.write(str(lens) + '\n')'''
    else:
        vi_word = {}
        with open("./wiki_data/pos_1/vi_pos.json", 'r', encoding='utf8') as file:
            vi_word = json.load(file)
        vi_line = []
        with open("./wiki_data/pos_1/vi.line", 'r', encoding='utf8') as file:
            for line in file.readlines():
                vi_line.append(int(line.rstrip()))
        '''en_word = {}
        with open("./wiki_data/pos/en_pos.json", 'r', encoding='utf8') as file:
            en_word = json.load(file)
        en_line = []
        with open("./wiki_data/pos/en.line", 'r', encoding='utf8') as file:
            for line in file.readlines():
                en_line.append(int(line.rstrip()))'''
    time2 = time.time()
    print(time2 - time1)
    en_vec = calc_vec(vi_word, vi_line)
    with open('./wiki_data/sem_vec/en.vec.json', 'w', encoding='utf8') as file:
        file.write(json.dumps(en_vec))

    
    #calc_Lij(zh_word, zh_line)

if __name__ == '__main__':
    #eval_zh()
    #eval_en()
    init_Lij(True)