
import os
import psutil
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # Importing tqdm for the progress bar

from codecon.utils.FileReader import FileReader
from codecon.utils.CustomDataset import CustomDataset


def cl_nlp_pred(data_pred, model_path=None, language = 'chn', benchmark = 0, mode = 'timefirst', batch_size=None):
    # Model name selection
    if language.lower() == 'chn':
        if mode.lower() == 'timefirst':
            model_name = 'hfl/chinese-roberta-wwm-ext'
        elif mode.lower() == 'qualityfirst':
            model_name = 'hfl/chinese-roberta-wwm-ext-large'
    elif language.lower() == 'eng':
        if mode == 'timefirst':
            model_name = 'google-bert/bert-base-uncased'
        elif mode == 'qualityfirst':
            model_name = 'google-bert/bert-large-uncased'

    print(f"选择的预训练模型为：{model_name}")

    if batch_size is None:
        if torch.cuda.is_available():
            # Estimate batch size based on GPU memory (simplified)
            gpu_properties = torch.cuda.get_device_properties(0)
            total_memory = gpu_properties.total_memory  # in bytes
            # Assuming each sample takes roughly 1MB (this is a simplification)
            batch_size = max(1, total_memory // (1e6 * 1024))  # Convert bytes to approximate units
            batch_size = min(batch_size, 64)  # Cap batch_size to prevent excessively large sizes
        else:
            # Use CPU cores to determine batch size
            cpu_cores = psutil.cpu_count(logical=True)
            batch_size = cpu_cores * 2  # Example strategy
            batch_size = min(batch_size, 32)  # Cap batch_size
        print(f"根据计算机配置自动设置batch size为 {batch_size}")

    data_loader = FileReader(data_pred=data_pred)
    df = data_loader.read_pred_file()
    df['label'] = 1
    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()

    # 分割数据集
    X_test = texts
    y_test = labels

    PRE_TRAINED_MODEL_NAME = model_name
    MAX_LENGTH = 512
    BATCH_SIZE = int(batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_path is None:
        directory = os.path.dirname(data_pred)
        model_path = os.path.join(directory, 'model')
        print(f"未指定模型路径，默认加载路径：{model_path}")
        model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    else:
        print(f"加载指定路径的模型：{model_path}")
        model = BertForSequenceClassification.from_pretrained(model_path).to(device)

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    test_dataset = CustomDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()

    true_labels = []
    pred_labels = []
    pred_prob = []
    count = 0

    benchmark = benchmark/100
    print(f"预测阈值设为：{benchmark}")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting", leave=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            max_prob, preds = torch.max(probabilities, dim=-1)
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()

            # 根据阈值更新预测标签列表
            for i in range(len(preds)):
                if max_prob[i] > benchmark:
                    pred_labels.append(preds[i])
                    pred_prob.append(max_prob[i])
                else:
                    pred_labels.append(-1)  # 将-1作为未知类别或其他指定值
            true_labels.extend(labels.cpu().tolist())

            count = count + 1
            if count % 100 == 0:
                print(f"已处理样本数：{count}")

    df['label'] = pred_labels
    directory = os.path.dirname(data_pred)
    output_path = os.path.join(directory, 'pred_results.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存至：{output_path}")

    return df

