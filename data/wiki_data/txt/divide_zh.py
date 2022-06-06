import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='init Lij')
    parser.add_argument('--number', help='the language to init')
    parser.add_argument('--language', help='the language to init')
    args = parser.parse_args()
    file = open("/home/zhfeng/data/wiki_data/txt/"+args.language+".all",'r',encoding='utf8')
    lines = file.readlines()
    number = 19084778
    out = open("/home/zhfeng/data/wiki_data/txt/"+args.language+".all",'w',encoding='utf8')
    weak_number = int(number*(float(args.number)/10))
    weak_number = min(len(lines), weak_number)
    for i in range(weak_number):
        out.write(lines[i])
