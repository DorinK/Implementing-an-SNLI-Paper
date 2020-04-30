README for assignment 4

***I used 'glove.6B.300d.txt' and SNLI data set, but couldn't submit them because they were too large***
			(Couldn't submit otherwise)

In the current working directory you should have:
1. Residual_stacked_encoder.py
2. Residual_stacked_encoder_test.py
2. utils.py
4. glove.6B.300d.txt
5. pre_trained_UNK (the pre trained vector for UNK words [took it from 'glove.840B.300d.txt'])
6. train set of SNLI ('snli_1.0_train.jsonl')
7. dev set of SNLI ('snli_1.0_dev.jsonl')
8. test set of SNLI ('snli_1.0_test.jsonl')


First run the whole training process and then check the test accuracy:
1. python3 Residual_stacked_encoder.py
  
* The modelFile and dictFile used in Residual_stacked_encoder_test.py will be outputted from 'Residual_stacked_encoder.py' to the current working directory.

2. python3 Residual_stacked_encoder_test.py
