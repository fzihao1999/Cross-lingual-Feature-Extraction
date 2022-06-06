import json
from re import L
import numpy as np
import os

if __name__=='__main__':
    print(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'crosslingual', 'dictionaries'))
    exit()
    x = np.zeros((3,4))
    y = np.zeros((1,3),dtype=int)
    y1 = np.zeros((1,3),dtype=int)
    x[0,0] = 3
    x[0,1] = 4
    x[0,2] = 5
    #x[0,3] = 9
    x[1,0] = 6
    x[1,1] = 2
    x[1,2] = 1
    #x[1,3] = 0
    x[2,0] = 3
    x[2,1] = 7
    x[2,2] = 6
    #x[2,3] = 8
    y[0,0] = 1
    y[0,1] = 2
    y[0,2] = 0
    y1[0,0] = 2
    y1[0,1] = 3
    y1[0,2] = 1
    #x-=y\
    print(np.multiply(0.5,x))
    xx = np.zeros((x.shape))
    print(xx)
    z = x
    z[0,0]=0
    print(x)
    print(z)
    zz = np.add(x , z)
    print(zz)
    #print(x.sort(axis=1))
    #print(x**2)
    #print(np.sum(x**2, axis=1))
    #print(len(x))
    #print(np.max(x,axis=1))
    #x = (x-np.mean(x, axis=1).reshape(3,1))/(np.max(x,axis=1) - np.min(x,axis=1)).reshape(3,1)
    #x -=  (np.mean(x, axis=1)) 

    '''ans = np.zeros(3)
    print(ans.shape)
    ans += x[y,y1].T
    print(ans)'''


    '''a=np.floor(10*np.random.rand(1,3))
    b=np.floor(10*np.random.rand(1,3))
    c=np.hstack((a,b))
    print(type(a))
    print(c)
    print(a)
    print(b)'''

    '''json_file = open("./wiki_data/sem_vec/vi.vec.json", 'r', encoding='utf8')
    words = {}
    words = json.load(json_file)
    out_file = open("./wiki_data/sem_vec/vi_sem.vec", 'w', encoding='utf8')
    number = len(words.keys())
    out_file.write(str(number) + " 300\n")
    for key in words.keys():
        vec = words[key]
        out_file.write(key + " ")
        vec_len = len(vec)
        for i in range(300):
            if i < vec_len:
                out_file.write(str(round(vec[i],6)) + ' ')
            else:
                out_file.write("0.0 ")
        out_file.write("\n")'''