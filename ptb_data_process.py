import codecs
import collections
import sys
from operator import itemgetter


RAW_TRAIN_DATA = "./data/ptb/ptb.train.txt"
RAW_TEST_DATA = "./data/ptb/ptb.test.txt"
RAW_EVAL_DATA = "./data/ptb/ptb.valid.txt"
VOCAB_OUTPUT = "./data/ptb/ptb.vocab"

OUTPUT_TRAIN_DATA = "./data/ptb/ptb.train"
OUTPUT_TEST_DATA = "./data/ptb/ptb.test"
OUTPUT_EVAL_DATA = "./data/ptb/ptb.valid"


def generate_vocab():
    counter = collections.Counter()
    with codecs.open(RAW_TRAIN_DATA, "r", "utf-8") as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    # 按词频顺序对单词进行排序
    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]
    sorted_words = ["<eos>"] + sorted_words

    with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
        for word in sorted_words:
            file_output.write(word + "\n")


def get_id(word, word_to_id):
    # 如果出现了被删除的低频词，则替换为"<unk>"
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]


def trans_word2num(in_file, out_file):
    with codecs.open(VOCAB_OUTPUT, "r", "utf-8") as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    fin = codecs.open(in_file, "r", "utf-8")
    fout = codecs.open(out_file, "w", "utf-8")
    for line in fin:
        words = line.strip().split() + ["<eos>"]
        out_line = " ".join([str(get_id(w, word_to_id)) for w in words]) + '\n'
        fout.write(out_line)
    fin.close()
    fout.close()


if __name__ == "__main__":
    trans_word2num(RAW_TRAIN_DATA, OUTPUT_TRAIN_DATA)
    trans_word2num(RAW_TEST_DATA, OUTPUT_TEST_DATA)
    trans_word2num(RAW_EVAL_DATA, OUTPUT_EVAL_DATA)
