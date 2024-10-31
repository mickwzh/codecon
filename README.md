[**🇨🇳中文**](https://github.com/mickwzh/codecon/edit/main/README.md) | [**🇬🇧English**](https://github.com/mickwzh/codecon/edit/main/README_eng.md)

<div align="center">
  <a href="https://github.com/mickwzh/codecon">
    <img src="https://github.com/mickwzh/codecon/blob/main/blob/codecon_logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# codecon: 为经济学研究者设计的机器学习魔术包

- [愿景](#愿景)
- [安装](#安装)
- [使用](#使用)
  - ⚠️环境配置
  - ⚠️数据准备
  - tp_nlp:基于无监督文本聚类的标签启发
  - cl_nlp_findtrain:基于文本相似度的训练样本扩展
  - cl_nlp_train,cl_nlp_pred:基于BERT的模型训练与预测
  - gai_nlp:批量调取生成式AI接口
- [python应急基础](#python应急基础)
- [联系](#联系)

## 愿景
### 用开源的方式帮助经济学研究者轻松实践机器学习方法,推动AI for Economics

#### _低代码_:尽可能减少代码量与计算机知识要求,根据需求自动挑选模型、配置最优参数
#### _为经济学而生_:参考经济学最新研究成果,设计符合经济学研究需要的功能
#### _清晰指南_:从python安装,服务器租借与使用,到算法深度介绍
#### _持续更新_:codecon将追踪产业界、学术界最新动态持续更新


## codecon v1.1 特点(26 Oct 2024)
经济学研究中大多数与文本分析相关任务都属于归类(Classification)问题。例如,情感分析(正向情感VS负向情感)、前瞻性分析(前瞻VS非前瞻)、是否与数字化转型相关(与数字化转型有关VS与数字化转型无关)。

Prof. Melissa Dell(2024) 提供了一个非常实用的经济学研究的文本分类任务实践流程,本次更新对该流程进行了完整实现与补充。

![blob/dell_2024_flowchart.png](https://github.com/mickwzh/codecon/blob/main/blob/dell_2024_flowchart.png)
(Dell, 2024)

简要来说,这张图重点讲了两个事情:

**第一,文本分类的机器学习实现主要有两个途径**
- _有监督学习_:标注少量数据后训练基于深度学习的文本分类器
- _生成式AI标注_:使用生成式AI,调整提示词,直接对文本进行标注

**第二,在使用有监督学习方法之前,可以借助词向量(Embeddings,[点击此处了解词向量](https://github.com/mickwzh/codecon/blob/main/note/embedding_note%20copy.ipynb))辅助启发、扩充标签**
- _标签启发_:使用词向量对文本进行聚类,启发标签
- _训练集扩充_:通过计算词向量之间的相似度(语意相似度)辅助拓展训练集

本次更新提供实现这两种功能的魔法命令
- 四行命令实现BERT文本分类器训练全流程(标签启发,训练集扩充、模型训练、模型预测)
- 批量调取生成式AI API接口进行文本分类

个人非常喜欢 Prof. Melissa Dell 的工作,严谨、实用、有深度, 另分享Prof. Melissa Dell 最新关于此篇深度学习如何应用在经济学研究中的的[文章笔记](https://github.com/mickwzh/codecon/blob/main/note/MelissaDell_2024_note.pdf)   
Dell, M. (2024). Deep learning for economists (No. w32768). National Bureau of Economic Research.


## 安装
### 强烈建议新建一个虚拟环境以运行codecon库。如果你很熟悉虚拟环境或租借服务器的步骤，可以直接跳过；如果你是初学者，请一定仔细阅读以下`使用-环境配置`的说明后后再进行安装。
```python
pip install codecon --upgrade -i https://pypi.org/simple
```
### 如果安装过于缓慢,可以打开全局梯子后重新pip，或者尝试

```
pip install codecon --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 或者通过github clone此项目
```python
git clone https://github.com/mickwzh/codecon.git
```

### 为了方便大家实践codecon, 准备了一个关于情感分类(二分类)的数据集供大家下载  

**百度网盘**: `https://pan.baidu.com/s/1dIrQQnOl6spZv6Jet48eQA?pwd=dvqm 提取码: dvqm`  
**Dropbox**: `https://www.dropbox.com/scl/fo/9ut3qtfradrde9tp6kslc/ANPvqvIgi2Lkf5t6TDuqiCc?rlkey=iw3seyzgpz9utmnj6xtdgej22&st=cywzuwmc&dl=0`

## 使用
### 环境配置
所谓环境，是指**一套**符合当前所需功能的工具包。
python的方便之处在于有大量封装好的工具包；但这些工具包往往依赖于其他的工具包。 
- 假设`A`和B两个python库分别使用的是`C1`和`C2`,
- `C1`和`C2`是`C`的两个不同版本，一个环境里通常只能安装一个`C`
- 使用`A`必须使用`C1`，使用B必须使用`C2`
- 所以如果先安装了`A`，就会导致安装B出错 (`C`的版本不兼容); 反之亦然。

如果你通过`pip`的方式安装`codecon`，其会自动配置好实现codecon的环境，但前提是你的旧环境中没有与codecon所依赖的工具包相冲突的工具包。  

**因此，建议在安装`codecon`前配置一个新环境。**

这里提供了适合初学者上手的两种（无代码）创建新环境的方案：[ANACONDA&Spyder(本地运行)](https://github.com/mickwzh/codecon/blob/main/note/anaconda_intro.pdf) 和 [Featurize&JupyterNote(在线运行)](https://github.com/mickwzh/codecon/blob/main/note/featurize_intro.pdf) 。点击可获取相应pdf版指南。  

⚠️ 如果个人电脑没有GPU, 或者无法保证长时间稳定运行，强烈建议[Featurize(服务器租借)](https://github.com/mickwzh/codecon/blob/main/note/featurize_intro.pdf)。

-----------------


### 数据准备
为尽可能减少使用者的工作量,`codecon`实践文本分类任务任务全流程需要且仅需要两个简单表格,请严格按照以下指南准备数据

- 在你喜欢的路径下新建一个文件夹,之后所有的结果与过程文件都将自动保存到该文件夹下
- 在该文件夹中准备好原表格(`raw_file`)与预测表格(`pred_file`),支持`.csv`,`.xls`,`.xlsx`,接下来以 `.xls` 格式为例

- `raw_file.xls` 为已经手工标记好的数据,包含 `text` 列和 `label` 列。`text`列为文本,`label`列为文本对应的标签(从0开始的连续整数)。
  - ⚠️ `label`必须为从0开始的连续整数 (0,1,2,...)。必须从0开始,必须为整数,必须连续。  
  - ⚠️ 列名必须为`text`和`label`,不能为其他列名

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

### 应用1: 基于无监督文本聚类的标签启发

🌟基于不同模型生成的文本向量对原始文本进行无监督聚类

#### STEP1: 新建一个文件夹,准备好将要被贴标签的预测表格(pred_file)
  - `pred_file.xls` 为之后需要在上面贴标签的数据,包含`text`列
  - 此步骤不需要`raw_file.xls`
#### STEP2: 运行代码,其中需要输入四个参数
  - `data_pred`: 输入pred_file的文件路径 (如何复制文件夹路径
  - `language`: 输入所需处理文本的语言, 中文输入 `'chn'`, 英文输入 `'eng'`
  - `method`: 输入聚类所用的模型,基于LDA方法输入`'LDA'`, 基于BERT深度学习方法输入 `'BERTopic'`
  - `n_topic`: 输入希望输出的聚类数。如果不输入,自动选择聚类数目

```python
import codecon
codecon.tp_nlp(data_pred = '替换为你的pred_file文件路径',
               language='chn', #'chn' 或 'eng'
               method='BERTopic', #'LDA' 或 'BERTopic'
               n_topic=None) #输入你希望模型生成的聚类数量
```
#### STEP3: 在`pred_file.xls`的文件夹下生成结果文件  
 - `labeled_data_pred_BERTopic.csv` / `labeled_data_pred_LDA.csv`: 对`pred_file.xls`中的每条文本贴类别标签 
 - `topics_description_BERTopic.csv` / `topics_description_LDA.csv`: 每一个类别的信息(类别的代表关键词,代表文本)

-----------------

### 应用2: 基于文本相似度的训练样本扩展

🌟基于不同模型生成的文本向量计算文本相似度,挑选高相似度样本帮助扩充训练集

#### STEP1: 新建一个文件夹,准备好将要被贴标签的预测表格(pred_file)和已经贴有标签的原表格raw_file
- 支持`raw_file.xls`中有多个类别,最终会分别输出从`pred_file.xls`中找到的每个类别的高相似度文本
- 假设`类别X`中有多条文本,此时`pred_file.xls`中某一条文本与`类别X`的相似度 = 与`类别X`中每一条文本相似度的平均值
  - ⚠️ `label`必须为从0开始的连续整数 (0,1,2,...)。必须从0开始,必须为整数,必须连续。  
  - ⚠️ 列名必须为`text`和`label`,不能为其他列名

#### STEP2: 运行代码,其中需要输入四个参数
  - `data_pred`: 输入`pred_file.xls`的文件路径
  - `data_raw`: 输入`raw_file.xls`的文件路径
  - `language`: 输入所需处理文本的语言, 中文输入 `'chn'`, 英文输入 `'eng'`
  - `method`: 输入计算相似度(生成文本向量)时基于的模型
    - `tfidf`: 基于`TF-IDF`算法计算的文本向量,文本数量较少时不建议使用
    - `word2vec`: 基于`word2vec`算法计算的文本向量,文本数量较少时不建议使用
    - `cosent`: 基于`bert`的改进算法 `cosent`计算的文本向量(推荐),适用于绝大多数场景
  - `threshold`: 设置被认定为高相似样本的门槛(相似度百分数)。该值越大,挑选越严苛,扩展的文本数量越少

```python
import codecon
codecon.cl_nlp_findtrain(data_pred = '替换为你的pred_file文件路径', 
                         data_raw = '替换为你的raw_file文件路径', 
                         language='eng', #'chn' 或 'eng'
                         method='cosent', #'tfidf' 或 'word2vec' 或 'cosent'
                         threshold=80) #0-100的任意数
```

#### STEP3: 在保存`pred_file.xls`的文件夹下生成结果文件
- `label_{1,2...}_Extended_Results.csv` 每个文件对应 `raw_file`中每一个类别的扩展样本
  - ⚠️ 该步骤仅能辅助人工拓展样本,再生成扩张样本后,注意人工甄选后再加入训练集
  - ⚠️ 虽然`cosent`模型通常有更好的效果,但当文本类别的划分主要依靠其中的某些关键词划分时,`tfidf`, `word2vec`可能有更出色的效果

-----------------

### 应用3: 基于`BERT`的模型训练与预测

🌟从已标记好的`raw_file.xls`中选择20%的数据作为测试集合,用剩余80%的样本作为训练集合在预训练模型BERT上微调(fine tune),将微调好的模型在`pred_file.xls`做预测

#### STEP1: 新建一个文件夹,准备好将要被贴标签的预测表格(`pred_file.xls`)和已经贴有标签的原表格`raw_file.xls`
⚠️ `label`必须为从0开始的连续整数 (0,1,2,...)。必须从0开始,必须为整数,必须连续。  
⚠️ 列名必须为`text`和`label`,不能为其他列名

#### STEP2: 运行代码训练模型
- `data_raw`: 输入`raw_file.xls`的文件路径
- `language`: 输入所需处理文本的语言, 中文输入 `'chn'`, 英文输入 `'eng'`
- `imbalance`: 类别分布是否均匀(类别之间的数量是否悬殊,不知道就都试试)
- `mode`: 训练速度优先还是训练质量优先
- `epoch`: 训练的轮数,不一定越多越好。模型预设了一般情况下适用的参数。如有必要,可根据结果中的loss变化图选择。
- `batch_size`: 每次喂给机器学习的学习样本,理论上该数量越多训练越平滑。但是其取决于硬件条件,`codecon`会根据您的电脑配置自动选择一个合适的值(通常为6~12)。如果模型训练当中提示内存不足,需要手动调低此值。

⚠️ 如果运行中显示 ''GPU不可用'' 强烈建议租服务器完成该步骤, 否则训练速度很慢([如何租服务器](#租借服务器))

```python
import codecon
codecon.cl_nlp_train(data_raw = '替换为你的raw_file文件路径',
                     language='chn', #'chn' 或 'eng'
                     imbalance = 'balance', #'balance' 或 'imbalance'
                     mode = 'timefirst', #'timefirst' 或 'qualityfirst'
                     epoch=None, #可不填写。如有需求填写正整数。
                     batch_size=None) #可不填写。如有需求填写正整数。
```

#### STEP3: 查看模型训练结果
- 模型训练结束后自动在`data_raw.xls`所在的文件夹下保存以下几个文件
  - `train_confusion_matrix.png`:函混淆矩阵
  - `train_model_performance.txt`: 召回率/准确率/f1分数
  - `train_test_label.csv`: 20%测试集上的原标签与预测标签,帮助发现哪一类文本模型难以区分
  - `model`: 训练好的模型参数,之后的步骤直接调用,不需要进行任何操作

#### STEP4: 使用训练好的模型在剩余样本上做预测
- `benchmark`: 分类的本质是模型在不同的类别上预测概率,输出概率最大的那一类。但在经济学研究的文本分析任务中通常无法预设所有类别。 例如,假设一个二分类问题,模型给定`文本X`属于`type1` 49%的概率,`type2` 51%的概率,判定`文本X`属于`type2`。但此时实际上 `type1`不应该归为`type1`,与`type2`的任何一类。  
  - `benchmark = 0` 表示对所有样本进行强制分类；`benchmark = 80` 意味着只有当模型认为一个样本的属于某一类别的概率大于80%时才判定其属于该类别(模型的置信度为80%)
  - 如果模型在所有类别上给定的概率都小于`benchmark`,将该样本标记为 `-1` 类 (`labels = -1`)

```python
import codecon
codecon.cl_nlp_pred(data_pred = '替换为你的pred_file文件路径',
                    model_path = None,  #若pred_file与上一步中的模型在同一个文件夹下,无需填写；否则需填写model的保存路径
                    language = 'chn', #'chn' 或 'eng' (与STEP2保持一致即可)
                    benchmark = 0, #输入0-100的整数 
                    mode = 'timefirst',  #'timefirst' 或 'qualityfirst' (与STEP2保持一致即可)
                    batch_size=None) #可不填写。如有需求填写正整数。
```
- 在`pred_file.xls`的同一文件夹下生成 `pred_results.csv`

-----------------

### 应用4: 批量调取生成式AI接口

🌟使用Kimi大模型作为接口,使用前需在kimi平台申请密钥(很简单,步骤如下),`codecon`不收取任何费用  

#### STEP1: 申请kimi密钥
- 在kimi[官方开发者平台](https://platform.moonshot.cn/console/account)注册账号登陆
- 在左侧任务栏中点击实名认证并完成认证
- 点击左侧账户充值
  - 基础模型每处理/生成 100万 tokens (约为150-200万汉字)仅需12RMB
  - 查看不同模型的[收费标准](https://platform.moonshot.cn/docs/pricing/chat#%E8%AE%A1%E8%B4%B9%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5)
  - Kimi所发布的该系列模型的区别在于它们可以输入和输出的最大上下文长度,效果上基本没有区别
- 在坐车任务栏中点击API Key管理,点击屏幕右侧新建,随便起个名字
- 按照指示复制密钥到命令的key中,

### STEP2: 在pred_file上直接调用大模型
- `pred_file.xls` 为之后需要在上面贴标签的数据,包含`text`列
- `model` 根据输入输出文本长度的需要,选择`moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-128k`三种模型
- `task`: 输入你对任务的要求和描述,之后会对`pred_file.xls`中的每一行文本执行该任务
  - 例如,我将输入一段金融新闻,判断该新闻属于乐观语调还是悲观语调。不输出任何其他内容。
  - 例如,我将输入一段关于某公司高管的描述。按照姓名、年龄、就职公司、工作内容进行信息提取。不输出任何其他内容。  

⚠️如何设计任务描述除了尽可能结构化、步骤化没有技巧可言。建议先对少部分样本进行测试,对结果满意后再大批量调取。

```python
import codecon
codecon.gai_nlp(data_pred = '替换为你的pred_file文件路径',
                model="moonshot-v1-8k", #'moonshot-v1-8k', 'moonshot-v1-32k' 或 'moonshot-v1-128k'
                key="替换为STEP1中你得到的API Key",
                task="defaul task") #输入你对任务的描述
```

- kimi的标记结果 `label_gai_Results.csv` 将输出到与`pred_file.xls`相同的文件夹下,除了运行后全部保存之外,运行中间每10条保存一次
- 如果运行中出现错误(断网、余额不足...) 请检查最新的一版本 `label_gai_Results.csv`,从上次运行的断点处重新开始运行该命令


-----------------

## python应急基础
- 如果你是python小白却深陷RA的ddl,或者需要立即使用`codecon`搬砖,或许以下几步可以帮到你
- 如果是需要用`cl_nlp_train`,`cl_nlp_pred`训练大模型,强烈推荐直接租借服务器
### 本地运行
点击[**此处**](https://www.spyder-ide.org/)安装简单易用的[**Spyder**](https://www.spyder-ide.org/)
### 租借服务器
- 推荐[Featurize](https://featurize.cn/vm/available),注册账号,充一点钱(新注册会送代金券)
- 按小时租借RTX 4090一枚(1.87/小时)
- 选择镜像 (`PyTorch`版本较新的环境),点击开始使用  
  ⚠️不要选择`App市场`, 否则不能保证环境正常配置
- 在 我租用的实例中 点击打开工作区,点击左上角蓝色按钮,新建`Jupyter Notebook`
- 在第一行敲下(注意加感叹号),安装成功后使用即可
```python
!pip install codecon
```
- 进入Featurize的工作界面后,可直接从本地拖拽文件上传服务器,但是大文件建议从Featurzie首页数据集处上传,再回到工作区下载
### [一些python基本概念](https://github.com/mickwzh/codecon/blob/main/note/%E5%86%99%E4%B8%8B%E7%AC%AC%E4%B8%80%E8%A1%8C%E4%BB%A3%E7%A0%81%E5%89%8D%E9%9C%80%E8%A6%81%E7%9F%A5%E9%81%93%E7%9A%84%E4%BA%8B%20copy.md)
### [笔者关于大数据技术应用于经济学研究中的一些思考](https://github.com/mickwzh/codecon/blob/main/note/DataScienceAndSocialScience%20copy.md)

## 贡献
项目代码还很粗糙,如果大家对代码有所改进,欢迎提交回本项目,在提交之前,注意以下两点:

- 在`tests`添加相应的单元测试

之后即可提交PR。

## 联系
- 邮箱: mickwang@connect.hku.hk


