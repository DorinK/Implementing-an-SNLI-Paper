import torch
import numpy as np
from Residual_stacked_encoder import InferSentences, read_snli_file, words_and_tags_to_indexes, get_batch, \
    extract_from_batch, pad_sentences, BATCH_SIZE, DICT_FILE, MODEL_FILE, DEVICE


def accuracy_on_test_set(model, data_set):

    good = total = 0.0

    # Make batches
    loader = []
    for sample in get_batch(iter(data_set), BATCH_SIZE):
        loader.append(sample)

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

            total += tags.shape[0]

            # For each prediction and tag of an example in the batch
            for y_hat, tag in np.nditer([predictions, tags.numpy()]):

                if y_hat == tag:
                    good += 1

    # Calculating the loss and accuracy rate on the data set
    return good / total


def main():

    # Loading dictionaries
    dict = torch.load(DICT_FILE)

    w2i = dict['word_to_index']
    t2i = dict['tag_to_index']

    # Redefining the model
    model = InferSentences(vocab_size=len(w2i), output_size=len(t2i), embedding_dim=300, bilstm_output_dim=600,
                           hidden_layer_size=800, dropout=0.1)

    # Loading the state dictionary of the model
    state_dict = torch.load(MODEL_FILE)
    model.load_state_dict(state_dict)
    model.eval()

    if torch.cuda.is_available():
        model.cuda(DEVICE)

    print("Start reading the test file")
    test_data = read_snli_file("./snli_1.0_test.jsonl")
    print("Finished reading the test file\n")

    # Process test data
    processed_test_data = words_and_tags_to_indexes(test_data, w2i, t2i)

    print("Test validation")
    test_accuracy = accuracy_on_test_set(model, processed_test_data)
    print("Finished test validation\n")

    print("Test accuracy is: {}".format(test_accuracy))


if __name__ == '__main__':
    main()
