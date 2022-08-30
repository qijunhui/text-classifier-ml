import os


def makedir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)


# 根目录
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# 数据
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "takeaway_comment_8k.csv")

# 模型
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
LABELS_PATH = os.path.join(OUTPUTS_DIR, "labels.pkl")
WORD_VECTOR_PATH = os.path.join(OUTPUTS_DIR, "word_vector.pkl")
MNB_MODEL_PATH = os.path.join(OUTPUTS_DIR, "mnb_model.pkl")
LR_MODEL_PATH = os.path.join(OUTPUTS_DIR, "lr_model.pkl")
SVM_MODEL_PATH = os.path.join(OUTPUTS_DIR, "svm_model.pkl")

# makedir
makedir(OUTPUTS_DIR)
