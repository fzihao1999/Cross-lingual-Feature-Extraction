import numpy as xp
import argparse
import sys
import embeddings

# python pca.py -i wiki.en.vec -o wiki.en.50.vec --start 1 --end 51
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Word embeddings dimension reduction')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input word embedding file (defaults to stdin)')
    parser.add_argument('-o', '--output', default=sys.stdout.fileno(), help='the output word embedding file (defaults to stdout)')
    parser.add_argument('--start', type=int, default=0, help='the starting index of the principle components')
    parser.add_argument('--end', type=int, default=300, help='the ending index of the principle components')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    args = parser.parse_args()

    # Read input embeddings
    f = open(args.input, encoding=args.encoding, errors='surrogateescape')
    words, matrix, dim = embeddings.read(f)
    print("matrix: ", matrix.shape) # (N, embedding_dim)

    # Perform dimension reduction
    matrix = matrix - matrix.mean(axis=0)
    cov = xp.cov(matrix.T) / matrix.shape[0] # np.cov 计算协方差
    #print("cov: ", cov.shape) # (embedding_dim, embedding_dim)
    w, v = xp.linalg.eig(cov) # np.linalg.eig 计算矩阵特征向量（返回的w是其特征值。特征值不会特意进行排序，返回的v是归一化后的特征向量）
    #print("w :", w.shape, "  v:", v.shape) # (embedding_dim, )   (embedding_dim, embedding_dim)
    idx = w.argsort()[::-1] # 这里是从右往左 # 切片：a[:] a[::] a[::-1] https://www.jianshu.com/p/15715d6f4dad
    w = w[idx] 
    v = v[:, idx]
    #h = v[:, args.start:args.end]
    #print("h:", h.shape) 
    matrix = matrix.dot(v[:, args.start:args.end])
    # Write normalized embeddings
    f = open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape')
    embeddings.write(words, matrix, f)
