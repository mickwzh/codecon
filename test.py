import codecon

codecon.tp_nlp(data_pred = '/Users/zhaohuiwang/Downloads/test/2425_test_pred.csv', language='chn', method='LDA', n_topic=None)
codecon.cl_nlp_findtrain(data_pred = '/Users/zhaohuiwang/Downloads/test/2425_test_pred.csv', data_raw = '/Users/zhaohuiwang/Downloads/test/2425_test_raw.xls', language='eng', method='word2vec', threshold=80)
codecon.cl_nlp_train(data_raw = '/Users/zhaohuiwang/Downloads/test/2425_test_raw.xls', language='eng', imbalance = 'imbalance', mode = 'timefirst', epoch=None, batch_size=None)
codecon.cl_nlp_pred(data_pred = '/Users/zhaohuiwang/Downloads/test/2425_test_pred.csv', model_path='/Users/zhaohuiwang/Downloads/test/model', language = 'chn', threshold = 0.6, mode = 'timefirst', batch_size=None)
codecon.gai_nlp(data_pred = '/Users/zhaohuiwang/Downloads/test/2425_test_pred.csv', model="moonshot-v1-8k", key="sk-XZgg3TqjDqhwPlAwRveBMy1kxEOBnXGmZmdtzt8VXyTJp4a7", task="一句话总结该段文本")

