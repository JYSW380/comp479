import collections
import glob
import sys
from project_1 import generate_tokens_pipeline
import math

k = 2
b = 0.1


def query_parser(query: str):
    return sorted(generate_tokens_pipeline(query))


def find_file_index(spliting_words, term):
    if len(spliting_words) == 1 and spliting_words[0] == "":
        return 0
    for i in range(len(spliting_words)):
        if i + 1 < len(spliting_words) and spliting_words[i] < term <= spliting_words[i + 1]:
            return i + 1
        elif i == 0 and term <= spliting_words[i]:
            return 0
        elif i + 1 == len(spliting_words) and term >= spliting_words[i]:
            return i + 1

    return len(spliting_words)


def get_document_by_rank(index_files: str, words: list, splits: list, N, l_avg, verbose=False, rank_method="-BM25"):
    res = []
    if len(words) == 0:
        return res
    a = words[0]
    file_index = find_file_index(splits, a)
    f = open(index_files[file_index], "r")
    line = f.readline().strip("\n")
    while line:
        if line.split("#####")[0] == a:
            res.append(line.rstrip("\n").split("#####")[1].split("-->"))
            if verbose:
                print("index: 0", line)
            break
        elif line.split("#####")[0] > a:
            break
        else:
            line = f.readline().strip("\n")

    for i in range(1, len(words)):
        b_posting = []
        b = words[i]
        line = f.readline().strip("\n")
        while line:
            if line.split("#####")[0] == b:
                b_posting = line.rstrip("\n").split("#####")[1].split("-->")
                if verbose:
                    print("index:", i, line)
                break
            elif line.split("#####")[0] > b:
                break
            else:
                line = f.readline().strip("\n")
                if line == '':
                    f.close()
                    next_file_index = find_file_index(splits, b)
                    if file_index == next_file_index:
                        print("[INFO] No postings for ", b)
                        b_posting = []
                        break
                    f = open(index_files[next_file_index], "r")
                    line = f.readline().strip("\n")
                    file_index = next_file_index
        res.append(b_posting)
    f.close()
    res = document_rank(res, N, l_avg, rank_method)
    if verbose:
        print("get document:", res)
    return res


def help():
    print("[Usage] python3 project_1_query.py -[mode] [query:string] -[v:verbose]")


def BM25(ld, tf, N, df, l_avg):
    print(ld, tf, N, df, l_avg)
    return math.log(N / df) * ((k + 1) * tf / (k * ((1 - b) + b * (ld / l_avg)) + tf))


def tf_idf(tf, N, df):
    # print(ld, tf, N, df, l_avg)
    return math.log(N / df) * (1 + math.log(tf))


def document_rank(parameters, N, l_avg, rank_method):  # [['url~200~1', 'url~100~2']]  #url length-of-document term frequency
    res = {}
    for postings in parameters:
        for sub_postings in postings:  # ['url~200~1', 'url~233~5']
            url, ld, tf = [x for x in sub_postings.strip("\n").split("~")]  # here return [url, 235, 7]
            ld = int(ld)  # url can not int so do ld = int(ld) and  tf = int(tf)
            tf = int(tf)
            df = len(postings)
            if rank_method == "-BM25":
                score = BM25(ld, tf, N, df, l_avg)
            else:
                score = tf_idf(tf, N, df)
            if res.get(url, None) is not None:
                res[url] = res[url] + score
            else:
                res[url] = score
    sorted_res = sorted(res.items(), key=lambda key_value: key_value[1], reverse=True)[0:100]
    print("score:", sorted_res)
    return [item[0] for item in sorted_res]


if __name__ == "__main__":
    f = open("spliting_word.txt", "r")
    spliting_words = f.readline().strip("\n").split("-->")
    N = int(f.readline().strip("\n").split(":")[1])
    l_avg = int(f.readline().strip("\n").split(":")[1])

    f.close()
    print("[INFO] Spliting_words: ", spliting_words)
    print("[INFO] System arguments: ", sys.argv)

    files = sorted(glob.glob("./concordia_indexs/*.txt"))

    try:
        rank_method = sys.argv[1]
        query = query_parser(sys.argv[2])
        if len(sys.argv) == 4 and sys.argv[3] == "-v":
            verbose = True
        else:
            verbose = False
        print("[INFO] Query: ", query)
        print("[INFO] rank method: ", rank_method)
    except IndexError:
        print("Error! Please check command input")
        help()
        sys.exit(1)

    if rank_method == "-BM25":
        res = get_document_by_rank(files, query, spliting_words, N, l_avg, verbose, rank_method)[0:100]
        print("Results url:", res)
        print("total count", len(res))
    else:
        res = get_document_by_rank(files, query, spliting_words, N, l_avg, verbose, rank_method)[0:100]
        print("Results url:", res)
        print("total count", len(res))