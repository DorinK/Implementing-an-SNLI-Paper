# Implementing-an-SNLI-Paper
Third assignment in Deep Learning course.

In this assignment, i implemented an "Natural Language Inference" model, based on the Stanford SNLI dataset.
The dataset page is https://nlp.stanford.edu/projects/snli/.

The paper I chose to implement from the leaderboard is "600D Residual stacked encoders" by Yixin Nie and Mohit Bansal, that was published in November 2017.
Link to the paper: https://arxiv.org/pdf/1708.02312.pdf

The model consists of two separate components, a sentence encoder and an entailment classifier. The sentence encoder encodes each of the two sentences -premise and hypothesis - into a vector representation and then the classifier makes a three-way classification based on the representation vectors combination to label the relationship between the premise and the hypothesis as that of entailment, contradiction, or neural.

In my attempt to replicate the paper's result, I achived:

* 91.0584% accuracy on the training set.

* 82.9303% accuracy on the dev set.

* 83.0415% accuracy on the test set.
