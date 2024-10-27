
import os
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from codecon.utils.FileReader import FileReader
from codecon.utils.FocalLoss import FocalLoss
from codecon.utils.CustomDataset import CustomDataset


def cl_nlp_train(data_raw, language = 'chn', imbalance = 'imbalance', mode = 'timefirst', epoch=None, batch_size=None):
    # Model name selection
    if language.lower() == 'chn':
        if mode.lower() == 'timefirst':
            model_name = 'hfl/chinese-roberta-wwm-ext'
        elif mode.lower() == 'qualityfirst':
            model_name = 'hfl/chinese-roberta-wwm-ext-large'
    elif language.lower() == 'eng':
        if mode.lower() == 'timefirst':
            model_name = 'google-bert/bert-base-uncased'
        elif mode.lower() == 'qualityfirst':
            model_name = 'google-bert/bert-large-uncased'
    print(f"选择的预训练模型为：{model_name}")

    if epoch is None:
        if imbalance.lower() == 'imbalance':
            epoch = 8
        elif imbalance.lower() == 'balance':
            if mode.lower() == 'timefirst':
                epoch = 16
            elif mode.lower() == 'qualityfirst':
                epoch = 24

        print(f"自动设置训练轮数为 {epoch}")

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

    print("正在加载原始数据...")
    data_loader = FileReader(data_raw=data_raw)
    df = data_loader.read_raw_file()
    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    print(f"数据集已划分为训练集和测试集，训练样本数：{len(X_train)}，测试样本数：{len(X_test)}")

    # 设置参数
    PRE_TRAINED_MODEL_NAME = model_name
    MAX_LENGTH = 512
    BATCH_SIZE = batch_size
    EPOCHS = epoch

    print("如果是首次使用该设置，可能需要下载预训练模型，这将花费一些时间...")
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    print("预训练模型已准备就绪")
    # 创建数据加载器
    train_dataset = CustomDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = CustomDataset(X_test, y_test, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("检测到GPU，可加速训练")
    else:
        print("未检测到GPU，训练可能会较慢，建议选择 'timefirst' mode 或在有GPU的设备或服务器上进行训练")

    # 加载预训练的BERT模型
    model = BertForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels=len(set(labels)),  # 设置分类类别数
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = FocalLoss().to(device)

    training_loss_values = []

    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        try:
            for i, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                if imbalance == 'imbalance':
                    loss = criterion(outputs.logits, labels)
                elif imbalance == 'balance':
                    loss = outputs[0]
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\n[错误] 训练时检测到内存不足，请尝试减小batch size。\n")
                torch.cuda.empty_cache()  # Clear CUDA memory to prevent subsequent errors
                return "训练因内存不足而中止，请调小batch size后重试。"
            else:
                raise e

        avg_train_loss = total_loss / len(train_loader)
        training_loss_values.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}")


    plt.figure(figsize=(8, 6))
    plt.plot(range(1, EPOCHS + 1), training_loss_values, marker='o', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt_path = os.path.join(os.path.dirname(data_raw), 'train_loss.png')
    plt.savefig(plt_path)
    plt.close()
    print(f"训练loss变化图已保存至{plt_path}")

    # 保存模型
    directory = os.path.dirname(data_raw)
    output_path = os.path.join(directory, 'model')
    model.save_pretrained(output_path)
    print(f'训练后的模型已保存至{output_path}')
    # Train model

    print("开始测试...")
    model.eval()

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            # logits = outputs[0]
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(preds)


    results_df = pd.DataFrame({
        'Text': X_test,
        'True Label': true_labels,
        'Predicted Label': pred_labels
    })
    test_results_path = os.path.join(directory, 'train_test_label.csv')
    results_df.to_csv(test_results_path, encoding = 'utf-8-sig')
    print(f"测试集预测结果已保存至 {test_results_path}")

    # Classification report
    report_str = classification_report(true_labels, pred_labels)

    # Accuracy
    accuracy = accuracy_score(true_labels, pred_labels)

    # Confusion matrix
    labels_raw = sorted(set(df['label'].tolist()))

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_raw, yticklabels=labels_raw)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    cm_file_path = os.path.join(directory, 'train_confusion_matrix.png')
    plt.savefig(cm_file_path)
    plt.close()
    print(f"混淆矩阵图已保存至 {cm_file_path}")

    # Save performance metrics
    performance_path = os.path.join(directory, 'train_model_performance.txt')
    with open(performance_path, 'w') as file:
        file.write("Classification Report:\n")
        file.write(report_str + "\n")
        file.write("Confusion Matrix:\n")
        file.write(str(cm) + "\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
    print(f"模型性能指标已保存至 {performance_path}")


    if accuracy >= 0.7:
        print(f"模型准确率为 {accuracy:.4f}，表现良好")
    elif accuracy >= 0.5 and accuracy <0.7:
        print(f"模型准确率为 {accuracy:.4f}，效果一般，请检查混淆矩阵弄清楚哪些样本容易混淆")
    elif accuracy < 0.5:
        print(f'模型准确率为 {accuracy:.4f}，效果较差，可能是样本区分难度较大')

    return model