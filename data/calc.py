import time
import numpy as np
import json
import argparse

def lower_bound(l, r, pos, target):
    ans = l
    while l <= r:
        mid = int((l+r)/2)
        if target > pos[mid]:
            ans = mid
            l = mid + 1
        else:
            r = mid - 1
    return ans

def upper_bound(l, r, pos, target):
    ans = r
    while l <= r:
        mid = int((l+r)/2)
        if target < pos[mid]:
            ans = mid
            r = mid - 1
        else:
            l = mid + 1
    return ans

global TIME1, TIME2, TIME3

def get_ij(pos_i, pos_j):
    #calculate every Lij of word_i and word_j

    i = 0
    j = 0
    len1 = len(pos_i)
    len2 = len(pos_j)
    #print(len1, len2)
    n = 0
    L = 0
    dis = []
    global TIME1, TIME2, TIME3
    while True:
        time1 = time.time()

        #i_line, i_pos = get_pos(pos_i[i])
        #j_line, j_pos = get_pos(pos_j[j])
        time2 = time.time()
        TIME1 += time2 - time1
        #print(i, j, len1, len2)
        time1 = time.time()
        #j = upper_bound(j, len2-1, pos_j, pos_i[i])
        #i = lower_bound(i, len1-1, pos_i, pos_j[j])
        while pos_i[i] > pos_j[j]:
            j += 1
            if j >= len2:
                break 
        if j >= len2:
            break
        o = 1
        while pos_i[i] < pos_j[j]:
            i += 1
            if i >= len1:
                o = 0
                break
        if i >= len1:
            break
        i -= o
        time2 = time.time()
        TIME2 += time2 - time1

        time1 = time.time()
        #print(test_line, test_pos, std_line, std_pos, calc_dis(test_line, test_pos, std_line, std_pos, linelen, word_len))
        if pos_j[j] > pos_i[i]:
            #print(test_line, test_pos, std_line, std_pos, calc_dis(test_line, test_pos, std_line, std_pos, linelen, word_len))
            #dis.append(calc_dis(i_line, i_pos, j_line, j_pos, linelen, word_len))
            #n += 1
            #L += calc_dis(i_line, i_pos, j_line, j_pos, linelen, word_len)
            dis.append(pos_j[j] - pos_i[i])
        time2 = time.time()
        TIME3 += time2 - time1
        i += 1
        j += 1
        if i >= len1 or j >= len2:
            break
    return dis

def combine_dis(dis_fwd, dis_bwd):
    #calculate the average distance of (word_i...word_j) and (word_j...word_i)
    #use the formula exp(-Lij/50) to calculate the average distance

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
    #find the words that appear most frequently 

    word_dict = {}
    for key in word_pos.keys():
        word_dict[key] = len(word_pos[key])
    word_sort = sorted(word_dict.items(), key=lambda item:item[1], reverse=True)
    word_list = []
    for i in range(10000):
        word_list.append(word_sort[i][0])
    return word_list

def calc_vec(word_pos, words):
    #calculate the semantic similarity of every word

    #words = list(word_pos.keys())
    word_number = len(words)
    word_vec = {}

    #use the top frequency word to calculate the semantic similarity
    topk = find_top(word_pos)
    for i in range(word_number):
        if words[i] not in word_pos.keys():
            continue
        time1 = time.time()
        global TIME1, TIME2, TIME3
        TIME1, TIME2, TIME3 = 0, 0, 0
        dis_i = []
        for j in range(10000):
            #print(j)
            dis_fwd = get_ij(word_pos[words[i]], word_pos[topk[j]])
            dis_bwd = get_ij(word_pos[topk[j]], word_pos[words[i]])
            dis_i.append(combine_dis(dis_fwd, dis_bwd))
        dis_i.sort(reverse=True)
        word_vec[words[i]] = dis_i[:300]

        time2 = time.time()
        #print(TIME1, TIME2, TIME3)
        print("word: ", words[i], "    time: ", str(time2-time1), "      ", str(i), "/", str(word_number))
        #exit()
    return word_vec

def eval_pos(path):
    word_dict = {}
    file = open(path,'r',encoding='utf8')
    line_number = 0
    line_len = []
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

def init_Lij(load_file, language):
    #initialize the semantic similarity of the given language
    
    time1 = time.time()
    print(load_file, language)
    if load_file == False:
        #evaluate the position of every word in the given txt and save it to the file


        vi_word, vi_line = eval_pos("./wiki_data/txt/vi.all")
        filename = language + "_pos.json"
        with open("./wiki_data/pos/"+filename, 'w', encoding='utf8') as file:
            file.write(json.dumps(vi_word))
        filename = language + ".line"
        with open("./wiki_data/pos/" + filename, 'w', encoding='utf8') as file:
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
        #load the position of every from the file

        en_word = {}
        filename = language + "_pos.json"
        with open("./wiki_data/pos/" + filename, 'r', encoding='utf8') as file:
            en_word = json.load(file)
        words = []
        filename = language + ".vec"
        with open("./wiki_data/vec/" + filename, 'r', encoding='utf8') as file:
            first_line = True
            for line in file.readlines():
                if first_line == True:
                    first_line = False
                    continue
                word = line.split(' ')[0]
                if len(word) == 0:
                    words.append(" ")
                else:
                    words.append(word)
    time2 = time.time()
    print(time2 - time1)

    en_vec = calc_vec(en_word, words)
    filename = language + ".vec.json"
    with open('./wiki_data/sem_vec/' + filename, 'w', encoding='utf8') as file:
        file.write(json.dumps(en_vec))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='init Lij')
    parser.add_argument('--language', help='the language to init')
    parser.add_argument('--loadfile', help='if you try to calc the lij of one language at the first time, it is False, otherwise, it is True')
    args = parser.parse_args()
    init_Lij(args.loadfile, args.language)