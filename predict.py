from utils import load_pkl, tokenizer
from configs import LABELS_PATH, WORD_VECTOR_PATH, MNB_MODEL_PATH, LR_MODEL_PATH, SVM_MODEL_PATH


LE = load_pkl(LABELS_PATH)
WORD_VECTOR = load_pkl(WORD_VECTOR_PATH)
MODELS = {
    "mnb": load_pkl(MNB_MODEL_PATH),
    "lr": load_pkl(LR_MODEL_PATH),
    "svm": load_pkl(SVM_MODEL_PATH),
}


def predict(text, model):
    X = WORD_VECTOR.transform([tokenizer(text)])
    y = LE.inverse_transform(MODELS[model].predict(X))[0]
    return y


if __name__ == "__main__":
    text = "今天的饭好好吃哦！"
    print("[MNB] 预测结果:", predict(text, model="mnb"))
    print("[LR ] 预测结果:", predict(text, model="lr"))
    print("[SVM] 预测结果:", predict(text, model="svm"))
