import os
from time import time
import codecs

import sklearn
import scipy.stats
from sklearn.metrics import make_scorer

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import joblib

from MeCabTokenizer import MeCabTokenizer

class CorpusReader(object):
    def __init__(self, path):
        with codecs.open(path, encoding='utf-8') as f:
            sent = []
            sents = []
            for line in f:
                if line == '\n':
                    sents.append(sent)
                    sent = []
                    continue
                morph_info = line.strip().split('\t')
                sent.append(morph_info)
        # 学習データ 9割 テストデータ 1割
        train_num = int(len(sents) * 0.9)
        self.__train_sents = sents[:train_num]
        self.__test_sents = sents[train_num:]

    def iob_sents(self, name): # train, test データの取り出し
        if name == 'train':
            return self.__train_sents
        elif name == 'test':
            return self.__test_sents
        else:
            return None

model_file_name = "./saved_model.pkl"

def get_character_types(string): # 形態素の文字種抽出
    def get_character_type(ch): # 文字単位の判定
        def is_hiragana(ch): # 平仮名の判定
            return 0x3040 <= ord(ch) <= 0x309F
        def is_katakana(ch): # カタカナの判定
            return 0x30A0 <= ord(ch) <= 0x30FF

        if ch.isspace(): return 'ZSPACE' # 半角スペース
        elif ch.isdigit(): return 'ZDIGIT' # アラビア数字
        elif ch.islower(): return 'ZLLET'  # アルファベット小文字
        elif ch.isupper(): return 'ZULET'  # アルファベット大文字
        elif is_hiragana(ch): return 'HIRAG'  # 平仮名
        elif is_katakana(ch): return 'KATAK'  # カタカナ
        else: return 'OTHER'  # 漢字・記号を含むその他文字
    # 対象形態素の文字種を決定
    character_types = map(get_character_type, string)
    character_types_str = '-'.join(sorted(set(character_types)))

    return character_types_str

def extract_pos_with_subtype(morph): # 品詞細分類の抽出
    # * より前方を取得
    idx = morph.index('*')

    return '-'.join(morph[1:idx])            

def word2features(sent, i): # 形態素の素性演算
    features = {
        'bias': 1.0,
    }
    # 素性追加
    def add_futures(index):
        word_index = i + index
        positon_str = ""
        if index > 0:
            positon_str = "+" + str(index) + ":"
        elif index < 0:
            positon_str = "-" + str(abs(index)) + ":"
        word = sent[word_index][0] # 形態素(表層)
        chtype = get_character_types(word) # 文字種
        postag = extract_pos_with_subtype(sent[word_index]) # 品詞情報
        features.update({
            positon_str + 'word' : word,
            positon_str + 'type' : chtype,
            positon_str + 'postag' : postag,
        })
    # i番目の形態素について素性を追加
    add_futures(0)
    if i >= 2:
        # 2文字前の文字情報を素性として入れる
        add_futures(-2)

    if i > 0: # 1文字前の文字情報を素性として入れる
        add_futures(-1)
    else:
        features['BOS'] = True

    if i < len(sent)-1: # 1文字後の文字情報を素性として入れる
        add_futures(1)
    else:
        features['EOS'] = True

    if i < len(sent)-2: # 2文字後の文字情報を素性として入れる
        add_futures(2)

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [morph[-1] for morph in sent]


def sent2tokens(sent):
    return [morph[0] for morph in sent]

if not os.path.exists(model_file_name):
    # コーパスデータロード
    # コーパスデータについて
    # 　MeCabで形態素解析した結果にBIOタグを付加した情報
    #   出自：https://github.com/Hironsan/IOB2Corpus
    t0 = time()
    print("Load Corpus ")
    t0 = time()
    c = CorpusReader('./data/hironsan.txt')
    train_sents = c.iob_sents('train')
    test_sents = c.iob_sents('test')
    duration = time() - t0
    print("done in %fs" % duration)

    # 形態素・ラベルをそれぞれ X,Yとして設定
    print("Create train Data ")
    t0 = time()
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    duration = time() - t0
    print("done in %fs" % duration)

    # 学習
    # 'lbfgs' -> Gradient descent using the L-BFGS method
    print("Execute train[x_train:{}]".format(str(len(X_train))))
    t0 = time()
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    duration = time() - t0
    print("done in %fs" % duration)
    # モデルの保存
    print("Save Model File")
    t0 = time()
    joblib.dump(crf, model_file_name) 
    duration = time() - t0
    print("done in %fs" % duration)
    # テスト
    print("Execute Predict[{}]".format(str(len(X_test))))
    t0 = time()
    y_pred = crf.predict(X_test)
    duration = time() - t0
    print("done in %fs" % duration)
    # 結果スコアリング
    labels = list(crf.classes_)
    labels.remove('O')
    f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    print("f1-score:" + str(f1))
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))
else:
    crf = joblib.load(model_file_name)
    sample_text = "アメリカの捜査当局はトランプ前大統領の顧問弁護士を務めたジュリアーニ氏の自宅などを捜索したと、複数のメディアが伝えました。"
    tokenizer = MeCabTokenizer()
    morphs = tokenizer.extract_words(sample_text)
    X = [sent2features(s) for s in [morphs]]
    pred = crf.predict(X)
    for i, morph in enumerate(morphs):
        if pred[0][i] != "O":
            print(morph[0] + ":" + pred[0][i])

