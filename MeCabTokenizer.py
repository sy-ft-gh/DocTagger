import MeCab
class MeCabTokenizer:
    def __init__(self, dictionary="mecabrc"):
        self.dictionary = dictionary
        self.tagger = MeCab.Tagger(self.dictionary)

    def extract_words(self, text):
        if not text:
            return []

        words = []

        node = self.tagger.parseToNode(text)
        while node:
            features = node.feature.split(',')
            if features[0] != "BOS/EOS":
                word = [node.surface]
                word.extend(features)
                words.append(word)

            node = node.next

        return words

if __name__ == '__main__':
    text = "アメリカの捜査当局はトランプ前大統領の顧問弁護士を務めたジュリアーニ氏の自宅などを捜索したと、複数のメディアが伝えました。"
    wd = MeCabTokenizer()
    print(wd.extract_words(text))
