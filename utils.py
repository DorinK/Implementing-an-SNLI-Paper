import json
import numpy as np
import matplotlib.pyplot as plt


def read_snli_file(file_name):
    sentences_and_tags = []

    with open(file_name, 'r') as file:
        for line in file:
            js = json.loads(line[:-1])

            # Ignore gold labels
            if js["gold_label"] == "-":
                continue

            # Take only the premise,hypothesis and tag of each example
            sentences_and_tags.append((js["sentence1"], js["sentence2"], js["gold_label"]))

    return sentences_and_tags


def read_glove_file(file_name, vec_dim=300):
    words_vocab = []
    pre_trained_vecs = []

    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.split(' ')
            word = line.pop(0)
            words_vocab.append(word)
            pre_trained_vecs.append(np.array(list(map(float, line))))

    with open('./pre_trained_UNK.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.split(' ')
            word = line.pop(0)
            words_vocab.append(word)
            pre_trained_vecs.append(np.array(list(map(float, line))))

    # Make numpy array
    pre_trained_vecs = np.asarray([np.zeros(vec_dim)] + pre_trained_vecs)
    return words_vocab, pre_trained_vecs


def create_vocab(data, pad_token='PAD', unknown_token='UNK'):
    data.remove(unknown_token)
    data = sorted(data)

    # Add the pad_token
    vocab = [pad_token, unknown_token] + data

    # Map each word to a unique index
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    return word_to_ix


def words_and_tags_to_indexes(data, word_to_idx, tag_to_idx, unknown_token='UNK'):
    return [([word_to_idx[word] if word in word_to_idx else word_to_idx[unknown_token] for word in premise],
             [word_to_idx[word] if word in word_to_idx else word_to_idx[unknown_token] for word in hypothesis],
             tag_to_idx[tag]) for premise, hypothesis, tag in data]


def get_batch(iterable, n=1):
    current_batch = []

    for item in iterable:
        current_batch.append(item)

        if len(current_batch) == n:
            yield current_batch
            current_batch = []

    if current_batch:
        yield current_batch


def extract_from_batch(batch):
    sent1, sent2, tags = [], [], []

    # Separate premises,hypotheses and tags
    for premise, hypothesis, tag in batch:
        sent1.append(premise), sent2.append(hypothesis), tags.append(tag)

    return sent1, sent2, tags


def pad_sentences(sentences):

    # Get the length of each sentence
    lengths = [len(sentence) for sentence in sentences]
    longest_sent = max(lengths)

    # Create an empty matrix with padding tokens
    features = np.zeros((len(sentences), longest_sent), dtype=int)

    for ii, sentence in enumerate(sentences):
        if len(sentence) != 0:
            features[ii, :len(sentence)] = np.array(sentence)

    return features, np.array(lengths)


def plot_graph(title, train_acc, dev_acc):
    plt.title(title + " over Epochs")

    ticks = [i for i in range(1, len(train_acc) + 1)]
    plt.plot(ticks, train_acc, label='Train Accuracy', color='purple')
    plt.plot(ticks, dev_acc, label='Dev Accuracy', color='teal')

    # x,y labels
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")

    ticks = [1, 2, 3, 4, 5]
    plt.xticks(ticks)
    plt.legend(frameon=False)
    plt.show()
