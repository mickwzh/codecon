# 大数据技术与经济学研究中的实验

物理学诺贝尔得主、邦加鼓演奏者费曼(Richard Phillips Feynman, 1918-1988)在他1964年的一场演讲中讲述了他认为的科学研究方法:

"第一步，猜测事情如何发展，并且推导出猜测的结果;\
第二步，与实验比较，验证或修正猜想。"

*(“In general, we look for a new law by the following process. First, we guess it ... Then we compute the consequences of the guess...compare it directly with observations to see if it works. If it disagrees with experiment, it’s wrong.”)*

现代经济学同样遵循 “提出理论-->实验-->验证理论“ 的实验方式，但与自然科学有所不同。各个步骤思路大体如下：

**提出理论**：所谓好的理论，就是在多次试验下，依然经得起验证的猜测。社会科学中每个学科都有“猜测”(发展理论)的方法。特别自上世纪Arrow等经济学家将数学方法广泛地引入经济学后，经济学理论有了更加严谨、清晰、可证伪的表述方式。

**实验**：与自然科学不同，经济学研究多为准实验(缺乏对照组与实验组的随机分配)。数据的收集、清理、变量的构建，都是经济学研究者实验设计的一部分，也决定了理论验证的可靠性；在得到结构化数据后，通常会通过相关性分析等方式理解结构化数据(例如，GPD、价格等)。但经济学家普遍缺少对非结构化数据(例如，文本、音频等)的观测手段。

**验证理论**：经济学家致力于推断因果关系。从相关关系到因果关系存在巨大的鸿沟，经济学可信性革命之后，经验研究成为经济学不可或缺的一部分，基于统计学发展的计量经济学为此提供了强大的武器库(其他学科的同学很难想象简单的回归分析被玩出了多少花样)。

一直以来，经济学的争议之一在于，作为以人、社会为研究对象的学科，其很难设计出足够严谨的实验，得出像物理学一样普适性强的理论。但不管这个问题的答案如何，毫无疑问的是，大数据时代的到来对经济学研究的每一个步骤都将产生影响：通过大数据训练的机器人逐渐成为研究对象，甚至可以帮助研究者提出假设(Ludwig & Mullainathan, 2024); 已有相当丰富的文章探讨了机器学习、大数据等方法对计量经济学的影响(洪永淼 & 汪寿阳, 2021; Athey & Imbens, 2019)。

**CODECON**现阶段将专注于为经济学家的**实验**设计提供解决方案。

类比于初中的生物实验，我们将帮助研究者完成样本收集(数据爬取、数据库建立...)、切片制作(变量构建、数据收集...)、调试显微镜(非结构数据分析、可视化...)，以及致力于以开源的精神，将上述工作技能普及给每一个经济学、金融学、管理学等相关研究者，助力社会科学发展。

如果你是一个贝叶斯主义者，会相信无论是自然科学还是社会科学，不存在绝对正确的理论，所有的观测都会对理论(主观概率)产生修正。重要的是，作为研究者，我们有责任尽可能从实验中获得更加完整、准确的观察，采用恰当的方法完成推断，让我们的工作成为知识的一部分。

以费曼的另一句话作为结尾，共勉。

“在科学上绝对诚实是何等重要；第一个原则是，你一定不要欺骗你自己——况且你是最容易被欺骗的人……只有做到不欺骗自己之后，才能做到不去欺骗其他科学家。”

*("The first principle is that you must not fool yourself—and you are the easiest person to fool. So you have to be very careful about that. After you've not fooled yourself, it's easy not to fool other scientists."*)\
\
\
\
Reference:\
Athey, S., & Imbens, G. W. (2019). Machine Learning Methods That Economists Should Know About. Annual Review of Economics, 11(1), 685–725. https://doi.org/10.1146/annurev-economics-080217-053433

Ludwig, J., & Mullainathan, S. (2024). Machine Learning as a Tool for Hypothesis Generation. The Quarterly Journal of Economics, qjad055. https://doi.org/10.1093/qje/qjad055

洪永淼, & 汪寿阳. (2021). 大数据, 机器学习与统计学: 挑战与机遇. 计量经济学报, 1(1), 17-35.

费曼对科学研究方法的阐述. https://youtu.be/OL6-x0modwY\