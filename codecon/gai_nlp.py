import pandas as pd
from openai import OpenAI
import time
from tqdm import tqdm  # Import tqdm for progress visualization
import os

from codecon.utils.FileReader import FileReader


def predict_text(text, role_system_content, model, client):
    # 设置系统角色和内容
    system_message = {
        "role": "system",
        "content": role_system_content
    }

    # 用户消息
    user_message = {
        "role": "user",
        "content": text
    }

    # 调用Kimi API
    completion = client.chat.completions.create(
        model=model,
        messages=[system_message, user_message],
        temperature=0.3,
    )

    return completion.choices[0].message.content

def gai_nlp(data_pred, model="moonshot-v1-8k", key="YOUR_API_KEY", task="defaul task"):
    # 初始化客户端
    data_loader = FileReader(data_pred)
    df = data_loader.read_pred_file()

    client = OpenAI(
        api_key=key,  # 替换为您的 API Key
        base_url="https://api.moonshot.cn/v1"
    )


    directory = os.path.dirname(data_pred)
    output_path = os.path.join(directory, f'label_gai_Results.csv')

    # main
    df['label'] = None
    count = 0
    progress_bar = tqdm(total=len(df), desc='Labeling Progress', unit='texts')

    print("开始调用大模型对文本进行标注...")
    for i, row in df.iterrows():
        if count % 10 == 0:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"GAI标注数据集已保存至 {output_path}")
        while True:
            try:
                df.at[i, 'label'] = predict_text(row['text'], task, model,client)
                count = count + 1
                print(count)
                progress_bar.update(1)
                time.sleep(0.2)
                break  # 处理成功，退出循环

            except Exception as e:
                print(f"发生异常: {e}")
                print("等待20秒后重试...")
                time.sleep(20)
    # Save the extended dataset
    print("文本标注完成")

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return df