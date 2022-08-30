import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import load_pkl, save_pkl, tokenizer
from configs import DATA_PATH, LABELS_PATH, WORD_VECTOR_PATH, MNB_MODEL_PATH, LR_MODEL_PATH, SVM_MODEL_PATH


def label_encoder(y, labels_path):
    le = LabelEncoder()
    labels = le.fit_transform(y).tolist()
    save_pkl(labels_path, le)
    print("category in labels:", le.classes_)
    return labels


def load_data():
    datasets = pd.read_csv(DATA_PATH)
    X = [tokenizer(text) for text in datasets["text"].tolist()]
    y = label_encoder(datasets["label"].tolist(), LABELS_PATH)
    return X, y


def get_word_vector(X_train, feature="count", retrain=False):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer

    if (retrain is False) and os.path.exists(WORD_VECTOR_PATH):
        word_vector = load_pkl(WORD_VECTOR_PATH)
    else:
        print("start training Word Vector...")
        if feature == "count":
            word_vector = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 1), binary=True)
        elif feature == "tfidf":
            word_vector = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 1), binary=True)
        else:
            raise Exception("select the correct feature method...")
        word_vector.fit(X_train)
        save_pkl(WORD_VECTOR_PATH, word_vector)
    return word_vector


def get_mnb_model(X_train, y_train, retrain=False):
    from sklearn.naive_bayes import MultinomialNB

    if (retrain is False) and os.path.exists(MNB_MODEL_PATH):
        mnb_model = load_pkl(MNB_MODEL_PATH)
    else:
        print("start training MNB model...")
        mnb_model = MultinomialNB(alpha=1, fit_prior=True)
        mnb_model.fit(X_train, y_train)
        save_pkl(MNB_MODEL_PATH, mnb_model)
    return mnb_model


def get_lr_model(X_train, y_train, retrain=False):
    from sklearn.linear_model import LogisticRegression

    if (retrain is False) and os.path.exists(LR_MODEL_PATH):
        lr_model = load_pkl(LR_MODEL_PATH)
    else:
        print("start training LR model...")
        lr_model = LogisticRegression(max_iter=200, n_jobs=-1)
        lr_model.fit(X_train, y_train)
        save_pkl(LR_MODEL_PATH, lr_model)
    return lr_model


def get_svm_model(X_train, y_train, retrain=False):
    from sklearn.svm import SVC

    if (retrain is False) and os.path.exists(SVM_MODEL_PATH):
        svm_model = load_pkl(SVM_MODEL_PATH)
    else:
        print("start training SVM model...")
        svm_model = SVC()
        svm_model.fit(X_train, y_train)
        save_pkl(SVM_MODEL_PATH, svm_model)
    return svm_model


def ensemble(y_pred_list, ratio=0.5):
    return [1 if sum(collection) >= len(y_pred_list) * ratio else 0 for collection in zip(*y_pred_list)]


def classification_evaluate(y_true, y_pred):
    from sklearn import metrics

    print("accuracy on test data:\t", metrics.accuracy_score(y_true, y_pred))
    print("precision on test data:\t", metrics.precision_score(y_true, y_pred, zero_division=0))
    print("recall on test data:\t", metrics.recall_score(y_true, y_pred, zero_division=0))
    print("f1 on test data:\t\t", metrics.f1_score(y_true, y_pred, zero_division=0))
    print(metrics.classification_report(y_true, y_pred, target_names=load_pkl(LABELS_PATH).classes_, zero_division=0))

    print(f"confusion matrix:\n{metrics.confusion_matrix(y_true, y_pred)}")


def run_train(X_train, y_train, retrain=False):
    word_vector = get_word_vector(X_train, retrain=retrain)
    X_train, y_train = word_vector.transform(X_train), y_train
    print(f"sample & features in the train data: {X_train.shape}")
    mnb_model = get_mnb_model(X_train, y_train, retrain=retrain)
    lr_model = get_lr_model(X_train, y_train, retrain=retrain)
    svm_model = get_svm_model(X_train, y_train, retrain=retrain)
    return word_vector, mnb_model, lr_model, svm_model


def run_test(X_test, y_test, word_vector, **models):
    y_pred_of_models = {}
    X_test, y_test = word_vector.transform(X_test), y_test
    print(f"sample & features in the test data: {X_test.shape}")
    for name, model in models.items():
        y_pred_of_models[name] = model.predict(X_test)
    y_pred_of_models["em"] = ensemble(y_pred_of_models.values())

    for name, y_pred in y_pred_of_models.items():
        print(f"{'=*= ' * 10}[{name:^11}] {'=*= ' * 10}")
        classification_evaluate(y_test, y_pred)
        print(f"{'=*= ' * 10}[ end_model ] {'=*= ' * 10}\n")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    word_vector, mnb_model, lr_model, svm_model = run_train(X_train, y_train, retrain=False)
    run_test(X_test, y_test, word_vector, mnb=mnb_model, lr=lr_model, svm=svm_model)
