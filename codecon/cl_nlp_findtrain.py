import pandas as pd
import os
import re
import numpy as np
from tqdm import tqdm  # Import tqdm for progress visualization
from codecon.utils.FileReader import FileReader
import jieba
import importlib.resources
import nltk
from nltk.corpus import stopwords

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec

nltk.download('punkt_tab')
nltk.download('stopwords')

def preprocess(text, language):
    if language == 'chn':
        tokens = jieba.lcut(text)
        tokens = [word.strip() for word in tokens if re.match(r'[\u4e00-\u9fff]+', word)]
        stop_words = set(stopwords.words('chinese')) if 'chinese' in stopwords.fileids() else set()
        tokens = [word for word in tokens if word not in stop_words]
    elif language == 'eng':
        tokens = nltk.word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    else:
        raise ValueError("语言参数必须为 'eng' 或 'chn'")
    return tokens


def cl_nlp_findtrain(data_pred, data_raw, language='chn', method='tfidf', threshold=80):

    print("开始加载预测数据和原始数据...")
    data_loader = FileReader(data_pred=data_pred, data_raw=data_raw)
    df_pred = data_loader.read_pred_file()
    df_raw = data_loader.read_raw_file()


    pred_texts = df_pred['text'].astype(str).tolist()
    raw_texts = df_raw['text'].astype(str).tolist()
    raw_labels = df_raw['label'].tolist()

    unique_labels = sorted(list(set(raw_labels)))

    if method.lower() == 'tfidf':
        print("使用TF-IDF向量化...")
        # TF-IDF Vectorization

        vectorizer = TfidfVectorizer(max_features=5000,
                                         tokenizer=lambda text: preprocess(text, language))

        # Progress bar for TF-IDF vectorization
        tqdm.pandas(desc="TF-IDF 向量化")
        all_texts = raw_texts + pred_texts
        tfidf_matrix = vectorizer.fit_transform(tqdm(all_texts, desc="TF-IDF Vectorization"))

        print("计算相似度...")
        # Calculate similarity
        raw_tfidf = tfidf_matrix[:len(raw_texts)]
        pred_tfidf = tfidf_matrix[len(raw_texts):]
        similarity_matrix = cosine_similarity(pred_tfidf, raw_tfidf)

    elif method.lower() == 'word2vec':
        print("训练Word2Vec模型...")
        # Train a Word2Vec model
        tokenized_texts = [preprocess(text, language) for text in tqdm(raw_texts + pred_texts, desc="Tokenization")]
        word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=300, window=5, min_count=1, workers=4)

        def text_to_vector(tokens):
            vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec_model.vector_size)

        print("使用Word2Vec进行向量化...")
        raw_embeddings = np.array([text_to_vector(preprocess(text, language)) for text in tqdm(raw_texts, desc="Raw Texts Vectorization")])
        pred_embeddings = np.array([text_to_vector(preprocess(text, language)) for text in tqdm(pred_texts, desc="Pred Texts Vectorization")])

        print("计算相似度...")
        similarity_matrix = cosine_similarity(pred_embeddings, raw_embeddings)

    elif method.lower() == 'cosent':
        print('正在加载预训练模型（首次运行可能需要较长时间）...')
        # Load "shibing624/text2vec-base-chinese"

        if language == 'eng':
            tokenizer = AutoTokenizer.from_pretrained("shibing624/text2vec-base-multilingual")
            model = AutoModel.from_pretrained("shibing624/text2vec-base-multilingual")
        elif language == 'chn':
            tokenizer = AutoTokenizer.from_pretrained("shibing624/text2vec-base-chinese")
            model = AutoModel.from_pretrained("shibing624/text2vec-base-chinese")

        print('预训练模型已准备完毕')
        print("使用cosent进行向量化...")
        # Tokenization and encoding with progress bar
        def encode_texts(texts, tokenizer, model, max_length=512):
            embeddings = []
            for text in tqdm(texts, desc="cosent向量化"):
                inputs = tokenizer(text, truncation=True, max_length=42, return_tensors='pt')
                with torch.no_grad():
                    outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()  # Use [CLS] token embeddings
                embeddings.append(embedding[0])
            embeddings = np.array(embeddings)
            return embeddings

        raw_embeddings = encode_texts(raw_texts, tokenizer, model)
        pred_embeddings = encode_texts(pred_texts, tokenizer, model)

        similarity_matrix = cosine_similarity(pred_embeddings, raw_embeddings)

    else:
        raise ValueError("无效的method，请选择 'tfidf', 'word2vec' 或 'cosent'。")

    print("扩展训练数据中...")
    extended_data = []
    if threshold is None:
        threshold = 80  # Set a threshold for similarity to select similar samples
    else:
        threshold = threshold

    for i, sim_row in enumerate(similarity_matrix):
        max_sim_idx = sim_row.argmax()
        max_sim_val = sim_row[max_sim_idx]
        if max_sim_val > threshold:
            extended_data.append({'text': pred_texts[i], 'label': raw_labels[max_sim_idx]})


    indices_dict = {num: [] for num in unique_labels}
    # Iterate through the numbers list
    for num in unique_labels:
        # Find indices of 'num' in target_list
        indices = [i for i, x in enumerate(raw_labels) if x == num]
        # Store the indices in the dictionary
        indices_dict[num] = indices

    for key, value in indices_dict.items():
        selected_columns = similarity_matrix[:, value]
        row_means = np.mean(selected_columns, axis=1)

        percentile = np.percentile(row_means, threshold)
        indices = np.where(row_means > percentile)[0]
        selected_texts = [pred_texts[i] for i in indices]
        df_extended = pd.DataFrame()
        df_extended['text'] = selected_texts
        df_extended['label'] = key

        # Save the extended dataset
        directory = os.path.dirname(data_raw)
        output_path = os.path.join(directory, f'label_{key}_Extended_Results.csv')
        df_extended.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"{key}类扩展数据集已保存至{output_path}")

    return df_extended
