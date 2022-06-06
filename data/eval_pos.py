import json

def get_pos(pos):
    pos = pos.split(" ")
    return int(pos[0]), int(pos[1])

if __name__ == '__main__':
    vi_word = {}
    with open("./wiki_data/pos_1/en_pos.json", 'r', encoding='utf8') as file:
        vi_word = json.load(file)
    vi_line = []
    vi_line_sum = []
    vi_line_sum.append(0)
    pre = 0
    with open("./wiki_data/pos_1/en.line", 'r', encoding='utf8') as file:
        for line in file.readlines():
            vi_line.append(int(line.rstrip()))
            pre += int(line.rstrip())
            vi_line_sum.append(pre)
    
    vi = {}
    for key in vi_word.keys():
        pos = []
        for i in vi_word[key]:
            i_line, i_pos = get_pos(i)
            now_pos = vi_line_sum[i_line] + i_pos
            pos.append(now_pos)
        vi[key] = pos
    
    with open("./wiki_data/pos/en_pos.json", 'w', encoding='utf8') as file:
        file.write(json.dumps(vi))