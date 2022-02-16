# —*- coding: utf-8 -*-


class Config(object):
    def __init__(self, word_embedding_dimension=100, word_num=20000,
                 epoch=2, sentence_max_size=40, cuda=False,
                 label_num=2, learning_rate=0.01, hidden_size=100, batch_size=1,
                 out_channel=100):
        self.word_embedding_dimension = word_embedding_dimension     # dimension dim for words
        self.word_num = word_num
        self.epoch = epoch                                           # 
        self.sentence_max_size = sentence_max_size                   # sentence length
        self.label_num = label_num                                   # the number of label for classification
        self.lr = learning_rate
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.out_channel=out_channel
        self.cuda = cuda
