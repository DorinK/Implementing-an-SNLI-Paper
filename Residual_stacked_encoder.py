import torch as torch
from torch import nn, optim
from torch.nn import functional as F
from utils import *

EPOCHS = 5
SEED = 16
BATCH_SIZE = 32
START_LR = 0.0003
MODEL_FILE = "./modelFile"
DICT_FILE = "./dictFile"
DEVICE = 2

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class InferSentences(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, bilstm_output_dim, hidden_layer_size, dropout):
        super(InferSentences, self).__init__()

        self.num_layers = 1
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.bilstm1_output_dim = bilstm_output_dim
        self.bilstm2_output_dim = bilstm_output_dim
        self.bilstm3_output_dim = bilstm_output_dim
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.drop = dropout

        # Embedding layer
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(p=self.drop)

        # bilstm layers
        self.bilstm1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.bilstm1_output_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.bilstm2 = nn.LSTM(
            input_size=self.embedding_dim + self.bilstm1_output_dim * 2,
            hidden_size=self.bilstm2_output_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.bilstm3 = nn.LSTM(
            input_size=self.embedding_dim + self.bilstm2_output_dim * 2,
            hidden_size=self.bilstm3_output_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Linear layers
        self.linear1 = nn.Linear(self.bilstm3_output_dim * 2 * 4, self.hidden_layer_size)
        self.linear2 = nn.Linear(self.hidden_layer_size, self.output_size)

    def run_through_bilstm(self, bilstm, sentences, sentences_len):

        sorted_sentences_len, idx_sort = np.sort(sentences_len)[::-1], np.argsort(-sentences_len)
        idx_unsort = np.argsort(idx_sort)

        # Sort sentences by length
        idx_sort = torch.from_numpy(idx_sort).cuda(DEVICE) if torch.cuda.is_available() else torch.from_numpy(idx_sort)
        sentences = sentences.index_select(0, idx_sort)

        # Run the batch through bilstm
        sentences_packed = nn.utils.rnn.pack_padded_sequence(sentences, list(sorted_sentences_len), batch_first=True)
        sentences_output, _ = bilstm(sentences_packed)
        sentences_output, _ = nn.utils.rnn.pad_packed_sequence(sentences_output, batch_first=True)

        # Unsort
        idx_unsort = torch.from_numpy(idx_unsort).cuda(DEVICE) if torch.cuda.is_available() else torch.from_numpy(idx_unsort)
        sentences_output = sentences_output.index_select(0, idx_unsort)

        return sentences_output

    def sentence_encoder(self, sentences, sentences_len):

        # Get embeddings
        embed = self.embeddings(sentences)

        # Run through bilstms
        output_layer1 = self.run_through_bilstm(self.bilstm1, embed, sentences_len)

        input_layer2 = torch.cat([embed, output_layer1], dim=2)
        output_layer2 = self.run_through_bilstm(self.bilstm2, input_layer2, sentences_len)

        input_layer3 = torch.cat([embed, output_layer1 + output_layer2], dim=2)
        output_layer3 = self.run_through_bilstm(self.bilstm3, input_layer3, sentences_len)

        # Raw max pooling
        max_pool = [output_layer3[sent_i, :sent_len, :].max(dim=0)[0] for sent_i, sent_len in
                    enumerate(list(sentences_len))]

        return torch.stack(max_pool)

    def forward(self, premises, hypotheses, premises_len, hypotheses_len):

        # Get sentences representation
        premises_max_pool = self.sentence_encoder(premises, premises_len)
        hypotheses_max_pool = self.sentence_encoder(hypotheses, hypotheses_len)

        # Applying 3 matching methods to extract relations
        x = torch.cat([premises_max_pool, hypotheses_max_pool, torch.abs(premises_max_pool - hypotheses_max_pool),
                       premises_max_pool * hypotheses_max_pool], dim=1)

        x = self.activation(self.linear1(x))

        # Using dropout to prevent overfitting
        x = self.dropout(x)

        # Fully connected layer
        x = self.linear2(x)

        return F.log_softmax(x, dim=-1)


def train(model, optimizer, train, dev):
    epochs_train_acc, epochs_dev_acc = [], []

    # Compute the dev accuracy before training
    dev_accuracy, _ = accuracy_on_data_set(model, dev)
    print("Checking the starting Dev accuracy, before training:\n"
          "Dev Accuracy: {:.6f}".format(dev_accuracy))
    epochs_dev_acc.append(dev_accuracy)

    best_accuracy = dev_accuracy
    torch.save(model.state_dict(), MODEL_FILE)

    lr = START_LR

    for epoch in range(EPOCHS):

        # Declaring training mode
        model.train()

        # Shuffle the data
        np.random.shuffle(train)

        # Make batches
        train_loader = []
        for batch in get_batch(iter(train), BATCH_SIZE):
            train_loader.append(batch)

        sum_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):

            # Prepare data
            premises, hypotheses, tags = extract_from_batch(batch)
            premises, premises_lengths = pad_sentences(premises)
            hypotheses, hypotheses_lengths = pad_sentences(hypotheses)

            premises = torch.from_numpy(premises)
            hypotheses = torch.from_numpy(hypotheses)
            tags = torch.from_numpy(np.array(tags))

            if torch.cuda.is_available():
                premises = premises.cuda(DEVICE)
                hypotheses = hypotheses.cuda(DEVICE)
                tags = tags.cuda(DEVICE)

            # Reset the gradients from the previous iteration.
            model.zero_grad()

            # Forward pass
            outputs = model(premises, hypotheses, premises_lengths, hypotheses_lengths)

            # Compute the negative log likelihood loss.
            loss = F.nll_loss(outputs, tags)
            sum_loss += loss.item()

            # Back propagation- computing the gradients.
            loss.backward()

            # Update the parameters
            optimizer.step()

        # Compute the loss on the training set in the current epoch.
        train_loss = sum_loss / len(train)

        # Compute the accuracy on the training set in the current epoch.
        train_accuracy, _ = accuracy_on_data_set(model, train)

        # Compute the loss and accuracy on the dev set in the current epoch.
        dev_accuracy, dev_loss = accuracy_on_data_set(model, dev)

        # Save the dev's loss and accuracy results.
        epochs_train_acc.append(train_accuracy)
        epochs_dev_acc.append(dev_accuracy)

        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            torch.save(model.state_dict(), MODEL_FILE)

        print("Epoch: {}/{}...".format(epoch + 1, EPOCHS),
              "Train Loss: {:.6f}...".format(train_loss),
              "Train Accuracy: {:.6f}".format(train_accuracy),
              "Dev Loss: {:.6f}...".format(dev_loss),
              "Dev Accuracy: {:.6f}".format(dev_accuracy))

        # Decay the learning rate every two epochs
        if (epoch + 1) % 2 == 0:
            lr /= 2
            for group in optimizer.param_groups:
                group['lr'] = lr
                print("It's epoch number {} and the new lr is {}".format(epoch + 1, group['lr']))

    return epochs_train_acc, epochs_dev_acc[1:]


def accuracy_on_data_set(model, data_set):

    # Declaring evaluation mode.
    model.eval()

    good = total = 0.0
    sum_loss = 0.0

    # Make batches
    loader = []
    for batch in get_batch(iter(data_set), BATCH_SIZE):
        loader.append(batch)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):

            # Prepare data
            premises, hypotheses, tags = extract_from_batch(batch)
            premises, premises_lengths = pad_sentences(premises)
            hypotheses, hypotheses_lengths = pad_sentences(hypotheses)

            premises = torch.from_numpy(premises)
            hypotheses = torch.from_numpy(hypotheses)
            tags = torch.from_numpy(np.array(tags))

            if torch.cuda.is_available():
                premises = premises.cuda(DEVICE)
                hypotheses = hypotheses.cuda(DEVICE)
                tags = tags.cuda(DEVICE)

            # Forward pass
            outputs = model(premises, hypotheses, premises_lengths, hypotheses_lengths)

            outputs = outputs.detach().cpu()
            tags = tags.cpu()

            # Get the indexes of the max log-probability
            predictions = np.argmax(outputs.data.numpy(), axis=1)

            # Compute the Cross Entropy loss
            loss = F.nll_loss(outputs, tags)
            sum_loss += loss.item()

            total += tags.shape[0]

            # For each prediction and tag of an example in the batch
            for y_hat, tag in np.nditer([predictions, tags.numpy()]):

                if y_hat == tag:
                    good += 1

    # Compute the loss and accuracy rate on the data set
    return good / total, sum_loss / len(data_set)


if __name__ == "__main__":

    print("Start reading the train file")
    train_data = read_snli_file("./snli_1.0_train.jsonl")
    print("Finished reading the train file\n")

    print("Start reading the glove file")
    words, vecs = read_glove_file("./glove.6B.300d.txt")
    print("Finished reading the glove file\n")

    w2i = create_vocab(words)
    t2i = {"entailment": 0, "neutral": 1, "contradiction": 2}

    # Saving the dictionaries.
    torch.save({
        'word_to_index': w2i,
        'tag_to_index': t2i,
    }, DICT_FILE)

    # Process training data
    processed_train_data = words_and_tags_to_indexes(train_data, w2i, t2i)

    print("Start reading the dev file")
    dev_data = read_snli_file("./snli_1.0_dev.jsonl")
    print("Finished reading the dev file\n")

    # Process dev data
    processed_dev_data = words_and_tags_to_indexes(dev_data, w2i, t2i)

    model = InferSentences(vocab_size=len(w2i), output_size=len(t2i), embedding_dim=300, bilstm_output_dim=600,
                           hidden_layer_size=800, dropout=0.1)

    # Loading pre-trained embeddings to embedding layer.
    model.embeddings.weight.data.copy_(torch.from_numpy(vecs))

    if torch.cuda.is_available():
        model.cuda(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=START_LR)

    print("Start training")
    train_acc, dev_acc = train(model, optimizer, processed_train_data, processed_dev_data)
    print("Finished training\n")

    # plot_graph("Accuracy", train_acc, dev_acc)
