[**🇨🇳中文**](https://github.com/mickwzh/codecon/edit/main/README.md)

<div align="center">
  <a href="https://github.com/mickwzh/codecon">
    <img src="https://github.com/mickwzh/codecon/blob/main/blob/codecon_logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# codecon: A Machine Learning Magic Package Designed for Economics Researchers

- [Vision](#Vision)
- [Installation](#Installation)
- [Usage](#Usage)
  - ⚠️ Environment Configuration
  - ⚠️ Data Preparation
  - tp_nlp: Labeling Inspiration Based on Unsupervised Text Clustering
  - cl_nlp_findtrain: Training Sample Expansion Based on Text Similarity
  - cl_nlp_train, cl_nlp_pred: Model Training and Prediction Based on BERT
  - gai_nlp: Batch Calls to Generative AI Interfaces
- [Python Basics for Emergencies](#Basic Python Essentials)
- [Contact](#Contact)

## Vision
### To help economics researchers easily apply machine learning methods through open-source, advancing AI for Economics

#### _Low-Code_: Minimize code and computing knowledge requirements, automatically selecting models and configuring optimal parameters as needed
#### _Designed for Economics_: Features tailored to meet the needs of economic research, referencing the latest findings in economics
#### _Clear Guidance_: From Python installation, server rental and usage, to in-depth algorithmic introductions
#### _Continuous Updates_: Codecon will keep track of the latest developments in industry and academia and update continuously



## codecon v1.1 特点(26 Oct 2024)
Most text analysis tasks in economic research fall under the classification problem. For instance, sentiment analysis (positive sentiment vs. negative sentiment), forward-looking analysis (forward-looking vs. non-forward-looking), and whether a topic is related to digital transformation (related vs. not related to digital transformation).

Prof. Melissa Dell (2024) provided a highly practical workflow for text classification tasks in economic research. This update fully implements and supplements that workflow.

![blob/dell_2024_flowchart.png](https://github.com/mickwzh/codecon/blob/main/blob/dell_2024_flowchart.png)
(Dell, 2024)

In brief, this chart highlights two points:

**First, there are two main approaches to implementing machine learning for text classification**
- _Supervised Learning_: Train a deep learning-based text classifier after labeling a small amount of data
- _Generative AI Labeling_: Use generative AI, adjusting prompts, to label text directly

**Second, before using supervised learning, embeddings can be used to assist in labeling inspiration and training set expansion**
- _Label Inspiration_: Use embeddings to cluster text, inspiring labels
- _Training Set Expansion_: Use semantic similarity between embeddings to expand the training set

This update provides magic commands to implement these two functions:
- Four commands to complete the full process of BERT text classifier training (label inspiration, training set expansion, model training, model prediction)
- Batch call to generative AI API interface for text classification

Personally, I really admire Prof. Melissa Dell's work—rigorous, practical, and deep. Additionally, here is a note on Prof. Melissa Dell's latest article on how deep learning applies to economic research [Article Note](https://github.com/mickwzh/codecon/blob/main/note/MelissaDell_2024_note.pdf)  
Dell, M. (2024). Deep learning for economists (No. w32768). National Bureau of Economic Research.



## Installation
### It is highly recommended to create a new virtual environment to run the codecon library. If you are familiar with setting up virtual environments or renting servers, you can skip ahead; if you are a beginner, please read the instructions under `Usage - Environment Configuration` carefully before installation.

```python
pip install codecon --upgrade -i https://pypi.org/simple
```
### If the installation is too slow, you can turn on a global proxy and rerun pip, or try

```
pip install codecon --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Or clone this project from GitHub
```python
git clone https://github.com/mickwzh/codecon.git
```

### To facilitate practicing with codecon, a sentiment classification (binary classification) dataset is prepared for download

**Baidu Netdisk**: `https://pan.baidu.com/s/1dIrQQnOl6spZv6Jet48eQA?pwd=dvqm`  Access Code: `dvqm`  
**Dropbox**: `https://www.dropbox.com/scl/fo/9ut3qtfradrde9tp6kslc/ANPvqvIgi2Lkf5t6TDuqiCc?rlkey=iw3seyzgpz9utmnj6xtdgej22&st=cywzuwmc&dl=0`

## Usage
### Environment Configuration
An environment refers to a set of tools meeting the required functionalities. One of Python’s conveniences is the large number of pre-packaged tools; however, these packages often depend on other packages.  
The convenience of Python lies in its many pre-packaged toolkits; however, these toolkits often depend on other toolkits.  
- Suppose two Python libraries, `A` and `B`, use `C1` and `C2` respectively,
- `C1` and `C2` are two different versions of `C`, and usually only one version of `C` can be installed in one environment.
- Using `A` requires `C1`, and using `B` requires `C2`.
- So if `A` is installed first, installing `B` will cause an error (due to incompatible versions of `C`); vice versa is true.

If you install `codecon` via `pip`, it will automatically configure an environment for `codecon`, provided your existing environment does not contain any packages conflicting with those required by `codecon`.  

**Therefore, it is recommended to configure a new environment before installing `codecon`.**

Here are two options suitable for beginners to create a new environment without coding: [ANACONDA & Spyder (local)](https://github.com/mickwzh/codecon/blob/main/note/anaconda_intro.pdf) and [Featurize & JupyterNote (online)](https://github.com/mickwzh/codecon/blob/main/note/featurize_intro.pdf). Click to access the respective PDF guides.

⚠️ If your computer does not have a GPU, or cannot ensure stable operation for long periods, Featurize (server rental) is strongly recommended [Featurize (server rental)](https://github.com/mickwzh/codecon/blob/main/note/featurize_intro.pdf).

-----------------


### Data Preparation
To reduce user workload as much as possible, `codecon`’s full workflow for implementing text classification tasks requires only two simple tables. Please strictly follow the guidelines below to prepare the data:

- 在你喜欢的路径下新建一个文件夹,之后所有的结果与过程文件都将自动保存到该文件夹下
- 在该文件夹中准备好原表格(`raw_file`)与预测表格(`pred_file`),支持`.csv`,`.xls`,`.xlsx`,接下来以 `.xls` 格式为例

- Create a new folder in a preferred path; all resulting and intermediate files will be automatically saved in this folder.
- Prepare the raw file (`raw_file`) and prediction file (`pred_file`) in this folder, supporting `.csv`, `.xls`, and `.xlsx` formats. `.xls` is used as an example.

- `raw_file.xls` contains manually labeled data, with `text` and `label` columns. `text` contains the text, and `label` contains corresponding labels (starting with 0 as consecutive integers).
  - ⚠️ `label` must start from 0 as consecutive integers (0,1,2,...). It must start from 0, be integers, and be consecutive.  
  - ⚠️ Column names must be `text` and `label`, without variation.


| text                     | label |
|--------------------------|-------|
| Apple is yummy           | 1     |
| My new apple phone sucks | 0     |
| Apple is a fruit         | 1     |
| ...                      | ...   |

-  pred_file 之后想要在上面贴标签的数据,包含 text 列

| text                              |
|-----------------------------------|
| Apple is red                      |
| Apple stock be in the worst       |
| Apple is more popular than banana |
| ...                               |
-----------------

### Application 1: Labeling Inspiration Based on Unsupervised Text Clustering

🌟Cluster raw text using text vectors generated from different models in an unsupervised manner.

#### STEP 1: Create a new folder and prepare the prediction table (pred_file) to be labeled.
  - `pred_file.xls` contains the data to be labeled, with a `text` column.
  - This step does not require `raw_file.xls`.
#### STEP 2: Run the code with four parameters.
  - `data_pred`: Enter the path to the pred_file.
  - `language`: Language of the text to process, enter `'chn'` for Chinese, `'eng'` for English.
  - `method`: Model to use for clustering, enter `'LDA'` for LDA-based methods or `'BERTopic'` for BERT-based deep learning.
  - `n_topic`: Desired number of clusters. If not entered, it will automatically select the cluster count.

```python
import codecon
codecon.tp_nlp(data_pred='replace with your pred_file path',
               language='chn', #'chn' or 'eng'
               method='BERTopic', #'LDA' or 'BERTopic'
               n_topic=None) #enter the desired number of clusters
```
#### STEP 3: Generate result files in the `pred_file.xls` folder  
 - `labeled_data_pred_BERTopic.csv` / `labeled_data_pred_LDA.csv`: Labels each text in `pred_file.xls` by category 
 - `topics_description_BERTopic.csv` / `topics_description_LDA.csv`: Information for each category (representative key

-----------------

### Application 2: Training Sample Expansion Based on Text Similarity

🌟Calculate text similarity using text vectors generated from different models, selecting high-similarity samples to expand the training set.

#### STEP 1: Create a new folder and prepare both the prediction table (pred_file) and the labeled raw file (raw_file).
- `raw_file.xls` can contain multiple categories, and the model will output high-similarity text for each category.
- Suppose there are multiple texts in `Category X`. In this case, the similarity between a text in `pred_file.xls` and `Category X` = the average similarity with each text in `Category X`.
  - ⚠️ `label` must start from 0 as consecutive integers (0,1,2,...). It must start from 0, be integers, and be consecutive.  
  - ⚠️ Column names must be `text` and `label`, without variation.

#### STEP 2: Run the code, entering four parameters.
  - `data_pred`: Enter the path to `pred_file.xls`.
  - `data_raw`: Enter the path to `raw_file.xls`.
  - `language`: Language of the text to process, enter `'chn'` for Chinese, `'eng'` for English.
  - `method`: Model for similarity calculation (to generate text vectors).
    - `tfidf`: `TF-IDF`-based vectors, not recommended for smaller text quantities.
    - `word2vec`: `word2vec`-based vectors, not recommended for smaller text quantities.
    - `cosent`: BERT-based algorithm `cosent` for text vectors (recommended), suitable for most scenarios.
  - `threshold`: Set the threshold for high similarity samples (similarity percentage). The higher this value, the stricter the selection, yielding fewer extended texts.

```python
import codecon
codecon.cl_nlp_findtrain(data_pred='replace with your pred_file path', 
                         data_raw='replace with your raw_file path', 
                         language='eng', #'chn' or 'eng'
                         method='cosent', #'tfidf' or 'word2vec' or 'cosent'
                         threshold=80) #integer from 0-100
```

#### STEP 3: Generate result files in the `pred_file.xls` folder
- `label_{1,2...}_Extended_Results.csv`: Each file corresponds to an extended sample for each category in `raw_file`.
  - ⚠️ This step can only assist in manually expanding the sample set; carefully review the extended samples before adding them to the training set.
  - ⚠️ While the `cosent` model generally performs well, if category distinctions rely heavily on specific keywords, `tfidf` or `word2vec` may yield better results.

-----------------

### Application 3: Model Training and Prediction Based on `BERT`

🌟Select 20% of the labeled data from `raw_file.xls` as the test set and use the remaining 80% as the training set to fine-tune a pre-trained BERT model. The fine-tuned model is then used to make predictions on `pred_file.xls`.

#### STEP 1: Create a new folder, and prepare the prediction table (`pred_file.xls`) and the labeled raw file (`raw_file.xls`).
⚠️ `label` must start from 0 as consecutive integers (0,1,2,...). It must start from 0, be integers, and be consecutive.  
⚠️ Column names must be `text` and `label`, without variation.

#### STEP 2: Run the code to train the model
- `data_raw`: Enter the path to `raw_file.xls`.
- `language`: Language of the text to process, enter `'chn'` for Chinese, `'eng'` for English.
- `imbalance`: Specify if category distribution is balanced or unbalanced (try both if unsure).
- `mode`: Choose between speed-focused or quality-focused training.
- `epoch`: Number of training epochs, not always better when higher. The model has preset parameters suitable for most cases. Adjust based on the loss curve in the results if needed.
- `batch_size`: The sample size per learning iteration. More samples generally yield smoother training but are limited by hardware. `codecon` will auto-select an optimal size (typically 6–12). If memory issues arise, reduce this value manually.

⚠️ If "GPU not available" appears during execution, renting a server is strongly recommended, as training will otherwise be very slow ([Server Rental Guide](#renting-a-server))


```python
import codecon
codecon.cl_nlp_train(data_raw='replace with your raw_file path',
                     language='chn', #'chn' or 'eng'
                     imbalance='balance', #'balance' or 'imbalance'
                     mode='timefirst', #'timefirst' or 'qualityfirst'
                     epoch=None, #optional, enter a positive integer if needed
                     batch_size=None) #optional, enter a positive integer if needed
```

#### STEP 3: Review Model Training Results
- Upon completion of training, the following files are automatically saved in the folder containing `data_raw.xls`:
  - `train_confusion_matrix.png`: Confusion matrix
  - `train_model_performance.txt`: Recall, accuracy, and f1 scores
  - `train_test_label.csv`: Original and predicted labels in the 20% test set, helping to identify which types of text the model struggles to classify
  - `model`: Trained model parameters, ready to be called in the following steps without additional actions

#### STEP 4: Use the Trained Model for Prediction on Remaining Samples
- `benchmark`: Classification fundamentally involves predicting the probability for each category, selecting the category with the highest probability. However, in economic text analysis tasks, not all categories can always be predetermined. For instance, in a binary classification problem, suppose the model assigns `Text X` a 49% probability for `type1` and 51% for `type2`, thus categorizing it as `type2`. Yet, in reality, `Text X` may not belong to either `type1` or `type2`.
  - `benchmark = 0` enforces classification for all samples; `benchmark = 80` means a sample is only categorized if the model assigns a probability greater than 80% to a specific category (confidence level of 80%).
  - If the model’s probability for all categories is below `benchmark`, the sample is labeled as `-1` (`labels = -1`).


```python
import codecon
codecon.cl_nlp_pred(data_pred='replace with your pred_file path',
                    model_path=None,  # omit if pred_file and model are in the same folder; otherwise specify the model path
                    language='chn', #'chn' or 'eng' (keep consistent with STEP 2)
                    benchmark=0, # enter an integer between 0-100
                    mode='timefirst',  #'timefirst' or 'qualityfirst' (keep consistent with STEP 2)
                    batch_size=None) # optional, enter a positive integer if needed
```
- Generates `pred_results.csv` in the same folder as `pred_file.xls`

-----------------

### Application 4: Batch Access to Generative AI Interface

🌟Use the Kimi large model as an interface. You must obtain an API key from the Kimi platform before use (simple steps below), and `codecon` does not charge any fees.  

#### STEP 1: Obtain the Kimi API Key
- Register and log in to the Kimi [Official Developer Platform](https://platform.moonshot.cn/console/account).
- Click "Real-name Authentication" in the left sidebar and complete verification.
- Click "Account Recharge" in the left sidebar.
  - The basic model costs only 12 RMB for processing/generating 1 million tokens (about 1.5–2 million Chinese characters).
  - View different models' [pricing standards](https://platform.moonshot.cn/docs/pricing/chat#%E8%AE%A1%E8%B4%B9%E5%9F%B9%E6%A6%82%E5%BF%B5).
  - The primary difference among the Kimi models is their maximum input-output context length; performance is almost identical.
- In the sidebar, click "API Key Management," and then click "New" on the right. Give it any name.
- Follow the instructions to copy the key and paste it into the command under `key`.

### STEP 2: Directly Call the Large Model on `pred_file`
- `pred_file.xls` contains the data to be labeled, with a `text` column.
- `model`: Select from `moonshot-v1-8k`, `moonshot-v1-32k`, or `moonshot-v1-128k` based on input-output length requirements.
- `task`: Enter a clear task description that will be executed on each line in `pred_file.xls`.
  - For example, "I will input a piece of financial news. Determine if the news has an optimistic or pessimistic tone. Output no other content."
  - For example, "I will input a description of a company's executive. Extract details such as name, age, employer, and job duties. Output no other content."

⚠️ Designing task descriptions involves structuring and step-by-step instructions. Testing a small sample first is recommended. If results are satisfactory, proceed with batch processing.


```python
import codecon
codecon.gai_nlp(data_pred='replace with your pred_file path',
                model="moonshot-v1-8k", #'moonshot-v1-8k', 'moonshot-v1-32k' or 'moonshot-v1-128k'
                key="replace with the API Key obtained in STEP 1",
                task="default task") # enter your task description
```

- The Kimi-labeled result `label_gai_Results.csv` will be output in the same folder as `pred_file.xls`. In addition to saving after completion, the process saves every 10 records.
- If an error occurs during execution (e.g., disconnection, insufficient balance), check the latest version of `label_gai_Results.csv` and restart the command from the last saved point.


-----------------

## Basic Python Essentials
- If you’re new to Python and facing RA deadlines, or need to start using `codecon` immediately, the following steps might help.
- For large model training with `cl_nlp_train` or `cl_nlp_pred`, renting a server is highly recommended.

### Local Setup
Click [**Here**](https://www.spyder-ide.org/) to install the beginner-friendly [**Spyder**](https://www.spyder-ide.org/).

### Renting a Server
- We recommend [Featurize](https://featurize.cn/vm/available). Register an account, and add a small amount of funds (new users get coupons).
- Rent an RTX 4090 GPU by the hour (1.87 RMB/hour).
- Choose a recent environment version with `PyTorch`, then click Start.  
  ⚠️ Avoid selecting from the `App Market`, as it may not ensure the correct environment configuration.
- Under "My Instances," click "Open Workspace," then click the blue button at the top left to create a new `Jupyter Notebook`.
- In the first line, type (with an exclamation point) and run it once installation is complete.
```python
!pip install codecon
```
- Once in Featurize's workspace, you can drag and drop files directly from your computer to upload them to the server. For larger files, it’s recommended to upload them via the Featurize homepage’s data section and then download them within the workspace.
### [Some Basic Python Concepts](https://github.com/mickwzh/codecon/blob/main/note/%E5%86%99%E4%B8%8B%E7%AC%AC%E4%B8%80%E8%A1%8C%E4%BB%A3%E7%A0%81%E5%89%8D%E9%9C%80%E8%A6%81%E7%9F%A5%E9%81%93%E7%9A%84%E4%BA%8B%20copy.md)
### [Thoughts on Applying Big Data Technology in Economic Research](https://github.com/mickwzh/codecon/blob/main/note/DataScienceAndSocialScience%20copy.md)

## Contributing
The project code is still in the early stages. If you have improvements, feel free to submit them back to the project. Before submitting, please note the following:

- Add relevant unit tests in `tests`.

Once done, you can submit a PR.

## Contact
- Email: mickwang@connect.hku.hk



