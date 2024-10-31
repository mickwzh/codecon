import pandas as pd
import re
import os

from tqdm import tqdm
from gensim import corpora, models
from bertopic import BERTopic
import jieba
import nltk
from nltk.corpus import stopwords

from codecon.utils.FileReader import FileReader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

nltk.download('punkt')
nltk.download('stopwords')

def tp_nlp(data_pred, language='chn', method='BERTopic', n_topic=None):

    dataloader = FileReader(data_pred = data_pred)
    df = dataloader.read_pred_file()
    texts = df['text'].astype(str).tolist()

    directory = os.path.dirname(data_pred)

    # 初始化预处理函数
    def preprocess(text):
        if language == 'chn':
            tokens = jieba.lcut(text)
            # 去除非中文字符和停用词
            tokens = [word.strip() for word in tokens if re.match(r'[\u4e00-\u9fff]+', word)]
            stop_words = set(stopwords.words('chinese')) if 'chinese' in stopwords.fileids() else set()
            tokens = [word for word in tokens if word not in stop_words]
        elif language == 'eng':
            tokens = nltk.word_tokenize(text.lower())
            # 去除非字母字符和停用词
            tokens = [word for word in tokens if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        else:
            raise ValueError("语言参数必须为 'eng' 或 'chn'")
        return tokens

    processed_texts = []

    if method == 'LDA':
        print("选择的主题模型方法为 LDA")
        if language == 'chn':
            print("正在对中文文本进行预处理...")
        elif language == 'eng':
            print("正在对英文文本进行预处理...")
        else:
            raise ValueError("language 参数必须为 'eng' 或 'chn'")

        for text in tqdm(texts, desc="预处理文本"):
            tokens = preprocess(text)
            processed_texts.append(tokens)

        # 创建字典和语料库
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]

        # 如果n_topic为None，尝试找到最佳主题数
        if n_topic is None:
            n_topic = 10
            print(f"未指定主题数量，默认设置为{n_topic}个主题...")

        print(f"正在训练LDA模型，主题数：{n_topic}...")
        lda_model = models.LdaModel(corpus=corpus,
                                    id2word=dictionary,
                                    num_topics=n_topic,
                                    random_state=100,
                                    update_every=1,
                                    chunksize=100,
                                    passes=10,
                                    alpha='auto',
                                    per_word_topics=True)


        topics = []
        for bow in tqdm(corpus, desc="分配主题"):
            topic = lda_model.get_document_topics(bow)
            # 选择概率最高的主题
            topic = max(topic, key=lambda x: x[1])[0]
            topics.append(topic)

        # 添加主题到原数据
        df['topic'] = topics

        # 保存标注后的数据
        labeled_file = "labeled_data_pred.csv"
        datapred_output_path = os.path.join(directory, 'labeled_data_pred_LDA.csv')
        df.to_csv(datapred_output_path, index=False, encoding='utf-8-sig')
        print(f"已保存标注后的数据到 {datapred_output_path}")

        # 获取每个主题的关键词
        print("获取每个主题的关键词...")
        topic_keywords = {}
        for t in range(n_topic):
            keywords = lda_model.show_topic(t, topn=10)
            keywords = [word for word, _ in keywords]
            topic_keywords[t] = keywords

        # 获取每个主题的最典型文本
        print("获取每个主题的最典型文本...")
        typical_texts = {}
        for t in range(n_topic):
            # 找到主题t中概率最高的文本
            max_prob = -1
            typical_text = ""
            for i, bow in enumerate(corpus):
                doc_topics = lda_model.get_document_topics(bow)
                prob = dict(doc_topics).get(t, 0)
                if prob > max_prob:
                    max_prob = prob
                    typical_text = texts[i]
            typical_texts[t] = typical_text

        # 创建主题描述数据
        topic_descriptions = []
        for t in range(n_topic):
            description = {
                'topic': t,
                'keywords': ", ".join(topic_keywords[t]),
                'typical_text': typical_texts[t]
            }
            topic_descriptions.append(description)

        topic_df = pd.DataFrame(topic_descriptions)
        topics_file = "topics_description.csv"
        description_output_path = os.path.join(directory, 'topics_description_LDA.csv')
        topic_df.to_csv(description_output_path, index=False, encoding='utf-8-sig')
        print(f"已保存主题描述到 {description_output_path}")

    elif method == 'BERTopic':
        print("选择的主题模型方法为 BERTopic")
        print("使用 BERTopic 进行主题建模...")
        # BERTopic no need for preprocess
        model = BERTopic(nr_topics=n_topic, language='english' if language == 'eng' else "multilingual")
        topics, probs = model.fit_transform(texts)
        df['topic'] = topics
        datapred_output_path = os.path.join(directory, 'labeled_data_pred_BERTopic.csv')
        df.to_csv(datapred_output_path, index=False, encoding='utf-8-sig')
        print(f"已保存标注后的数据到 {datapred_output_path}")

        # 获取主题信息
        print("获取主题信息...")
        topic_info = model.get_topic_info()
        topics_description = []
        for _, row in topic_info.iterrows():
            t = row['Topic']
            if t == -1:
                continue  # 跳过噪音主题
            keywords = model.get_topic(t)
            keywords = [word for word, _ in keywords[:10]]
            # 获取最典型的文本
            representative_docs = model.get_representative_docs(t)
            typical_text = representative_docs[0] if representative_docs else ""
            description = {
                'topic': t,
                'keywords': ", ".join(keywords),
                'typical_text': typical_text
            }
            topics_description.append(description)

        topic_df = pd.DataFrame(topics_description)

        description_output_path = os.path.join(directory, 'topics_description_BERTopic.csv')
        topic_df.to_csv(description_output_path, index=False, encoding='utf-8-sig')
        print(f"已保存主题描述到 {description_output_path}")

    else:
        raise ValueError("method 参数必须为 'LDA' 或 'BERTopic'")

    print("文本主题建模完成。")