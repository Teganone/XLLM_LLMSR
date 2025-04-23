# 基于大型语言模型的结构化推理方法研究


## 摘要


## 1. 引言



## 2. 方法

### 2.1 整体架构

我们的LLM-SR方法采用了模块化设计，包括三个主要组件：

1. **问题解析模块**：负责从问题文本中提取所有必要条件。
2. **思维链解析模块**：从Question和CoT文本中识别陈述和证据对。
3. **验证模块**：使用O3-mini-high模型，判断证据是否充分支持陈述。

每一个模块有多种实现机制。
针对parser：
1. in-context learning：prompt engineering + few-shot
2. finetune: 用o3-mini解析得到的question_parsing和cot_parsing，结合question和cot作为输入，输出为训练集对应的question_parsing和cot_parsing训练模型。
verifier:
1. 用reasoning model(o3-mini-high) 直接推理。
2. tool-augmented：将问题和statement-evidence pair表示为z3验证程序，用python程序验证。 针对生成的逻辑程序出错的问题，引入self-refiner model，将程序执行的语法错误, 常见的错误汇总及其对应修改等回调给llm，直到生成正确的logic program/达到最大重复次数。

![alt text](image-1.png)
(代码有待整理和集成)

prompt重点要提到规则定义这一块（从数据得到的insights）：
question主要分为三部分：1. 问题描述； 2. 条件限制； 3. options。 因此在prompt中加了xxxx表明如何提取conditions；
cot: cot提取出来的statement必须是根据conditions等推断，evidence和conditions相关；以及多步推理的问题。。 因此在prompt中加了xxxx。。

输出格式解析：用正则表达式等方式提取、校验输出的格式的正确性。

介绍一下finetune：
这样的好处是可以利用强大的o3-mini的推理能力，同时预处理得到的qp和cp让llama无需学习qp和cp的解析的规则（需要大量数据），而是....

然后再介绍一下z3的设计；；； 
自定义logic program 语法， code translator转化为程序，执行程序。。
self refinement等。。。 

（这一块每一块都详细展开说说）


### 3.2 实验结果

#### 3.2.1 不同方法的比较

我们比较了不同方法在测试集上的性能，结果如下：

| 方法 | Question_F1 | Statement_F1 | Statement_Evidence_F1 | Reasoning_F1 |
|------|-------------|--------------|------------------------|--------------|


#### 3.2.2 参数敏感性分析
这一块还有待继续优化分析（实验还需要重新整合一下）。。
我们分析了不同参数配置对各个子任务性能的影响：

1. **温度参数（Temperature）**：
2. **推理能力（Reasoning）**：

### 3.3 结果分析


#### 3.3.1 方法优势


#### 3.3.2 方法局限性


## 4. 结论与未来工作

### 4.1 结论


### 4.2 未来工作


## 参考文献
1. Logic-LLM: 
2. LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning
