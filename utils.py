import dill
import jieba


def save_pkl(filepath, data):
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


def load_pkl(filepath):
    with open(filepath, "rb") as fr:
        data = dill.load(fr)
    print(f"[{filepath}] data loading...")
    return data


def tokenizer(text):
    return " ".join(jieba.lcut(text))
