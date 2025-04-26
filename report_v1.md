SRV: Empowering Self-Verification of Small Language Models through Step-wise Reasoning and Verification

## Abstract

大语言模型展现了思考能力但是思考过程缺乏可解释性和可控制性。为了更详细地评估LLM的思考过程，本paper面向XLLM Workshop of ACL 2025的LLM for Structural Reasoning任务。深入研究LLM的逐步思考过程，提取每步思考论证和论点，验证每步思考的正确性，从而量化思考过程。我们的方法提出了两个组件：解析器和验证器。我们首先解析器对问题和思维链进行解析，提取出每个问题的所有条件，思维链的逐步思考，再用验证器对逐步思考加以验证。为了解析思考过程，我们用更强大的语言模型的少量数据来微调小模型；为了验证思考的正确性，我们还引入了确定性符号求解器对公式化的思考过程进行推理。我们在https://huggingface.co/datasets/shuyi-zsy/LLMSR/blob/main/llmsr/reference_Test_A.json的数据集进行测试，和基准相比，分别提高了xxx。~~此外，我们还发现了问题解析和思维链解析的差异，问题解析需要更确定性的答案，而思维链解析更发散。（本paper的重点放在讨论思维链上）~~我们的研究结果和方法为细粒度分析逐步思考提供了一条有效的途径。


## Introduction

大型语言模型在自然语言处理领域取得了显著进展，但在需要精确逻辑推理的任务中（尤其是具有多个约束或子问题或需要专业知识的情况）仍面临挑战。传统的思维链方法虽然提高了模型的推理能力，但是思维链可能不可靠，导致下游推理表现不佳。最近的研究等表明大语言模型（OpenAI）可以自我纠正其回答并不断迭代直到产生适合的答案，这逐渐成为纠正不合适的生成的新的范式。但是能够自我纠正的模型都是参数量非常庞大的模型，需要大量的数据训练。https://arxiv.org/pdf/2404.17140等论文提出了 a novel pipeline to
generate self-correction data from a small LM,
and subsequently fine-tune the model to be a
self-correcting reasoner，并得到 The self-correction performance is largely bottlenecked by the verifier rather than the refiner.即小语言模型需要强验证器来提高self-correct reasoning。但是，该论文中的自我验证聚焦于整个思维链，没有对思维链进行精细化的验证。因此，本篇论文旨在通过采用少量数据，探索小型语言模型的更强大的自我验证机制。我们首先prompt Llama-3-8B-instruct生成logiqa问题的思维链，然后细粒度解析问题的每个条件，思维链的每个论证步骤，并加以验证。对于解析器，我们分别探索了规则设置prompt小型语言模型生成解析，和大型语言模型生成少量解析样本来微调小型语言模型；对于验证器，除了prompt大语言模型推理，我们还引入了确定性符号求解器，利用LLM将问题和推理步骤转化为符号公式，用符号求解器对公式化问题进行推理。
所有这些探索都旨在探索更细粒度更精确的自我验证器，来提高小型语言模型的自我纠正和推理能力，同时也为过程奖励建模提供更细粒度的奖励建模提供可能。
Framework:

![alt text](image-2.png)

## Problem Formulation of Self-Verification
我们通过解析并验证逐步思考来增强自我纠正过程中的验证器。

**Step-wise Reasoning := Question Parser + CoT Parser**
我们将逐步推理分解为两部分：问题解析和思维链解析。对于问题，我们可以将问题看成问题描述、限制条件、查询、和选项四部分，让LLM解析出前三部分中所有的条件。对于思维链解析，我们让LLM生成解决问题过程中的chain-of-thought，针对这个思维链，我们解析出每一步推理中的论点和论据。考虑到逻辑推理问题的特点，论点主要由问题的条件推演而来，并且可能和选项相关；论据多为问题条件或者中间推理的结论。问题解析和思维链解析可以整合也可以单独解析，便于对每个过程灵活调整参数，提高解析准确率。

**Step-wise Verification := Verifier**
对于思维链解析中的每步推理，验证器都会验证其准确性，标记为True/False。A verifier, either the LM itself (intrinsic) or the external signal (extrinsic), then judges whether the step-wise statement could be decluded by the corresponding evidence. 如果验证器无法成功推断，会默认推理步骤正确。

解耦解析和验证，甚至解藕问题验证和思维链验证，相对于“一模型全做”的设计有巨大优势：首先，我们可以自由地参数化每个模块——例如，通过使用微调和少量样本提示模型。其次，它降低了训练每个模块的难度，因为模型只需要专注于一种能力，即解析或者验证。最后，它让确定性的验证器引入外部符号求解器成为可能。


## Methodology

### Parser
我们的方法分为两个阶段: prompt 大语言模型生成问题解析和组合解析，将得到的问题解析和思维解析结合问题和思维链作为输入，目标的问题解析和思维解析作为标签，用来监督微调模型。

- **Stage 1: Generating Question_Parsing and CoT_Parsing**
我们设置了三种解析器：问题解析器，思维链解析器，组合解析器，分别用来解析问题、思维链和同时解析二者。这样方便我们为每种解析器调参。解析过程可以灵活选择搭配这三种解析器。在prompt中，我们限制了对应的规则，要求输出的解析尽可能和原问题/思维链保持语义一致。我们还分别采用了one-shot作为例子格式化输出结构，few-shot来增加逻辑问题样例的丰富性。
对应的prompt详情参考appendix。

- **Stage 2: Supervised Fine-tuning of the Parser**
为了在少样本的情况下微调LLama-3-8B-Instruct得到有效的解析器，我们在问题和思维链的基础之上增加Stage1得到的问题解析和思维链解析作为输入，这样模型无需从零学习解析规则，而是学习如何改进已有的解析结果。对应的prompt详情参考appendix。

### Verifier
我们分别采用大语言模型推断或者确定性符号求解器辅助验证器验证逐步推理。

#### LLM Verifier
对于解析器解析得到的每个逐步推理论点和论据，我们利用问题和论点论据 prompt LLM 来直接判断论点是否由论据推断，正确为“true",错误为“false”。下面是system prompt，详细的user prompt参见appendix（给出一两个user_prompt的例子）
```txt
LLM_VERIFT_SYSTEM_PROMPT = 'Whether the "statement" can be deduced from the "evidence" logically, answer with only with True or False, do not output other contents.'
```
#### Z3-Augmented Verifier
受Logic-LM（引用）的启发，我们将问题表述为对应的符号求解器公式，用符号求解器求解得到确定性的验证。

我们自定义符号公式和自然语言处理的中间表示的语法规则，prompt大语言模型生成对应问题和逐步推理的逻辑表达式，再将中间表示翻译为可运行的程序，用对应的符号求解器求解，解释得到的结果为“true”/”false“。
The inputs of our model are a logical reasoning question Q described in natural language, along with the step-wise reasoning R in the form of statement-evidence pairs. Our goal is to verify each statement-evidence pair. The Z3-Augmented Verifier follows a paradigm including Problem Formulation and Symbolic reasoning to solve the problem.
- **Problem Formulation**
Given a natural language logical reasoning question Q and its step‐wise reasoning R, we prompt a LLM to translate them into self-defined intermediate representations, which are then translated into a formal, SMT‐compatible representation by code translator. This encoding captures both the question description and step-wise reasoning in a symbolic language understood by Z3.
- **Symbolic Reasoning**
We invoke Z3 as a deterministic SMT solver over the encoded problem. Z3 efficiently checks satisfiability, carries out the required logical inferences, and produces a symbolic answer. Because Z3’s algorithms are sound and complete for the supported theories, the correctness of answer is guaranteed whenever the initial encoding is faithful.
- **Self-Refiner**
对于复杂问题和复杂推理，大语言模型很难生成正确的逻辑表达式。因此，我们引入了自我反馈模块，将Z3求解器的语法错误回传给LLM，指导其生成正确的逻辑表达式,迭代修正直到生成有效程序或达到最大尝试次数。
- **Result Interpretation**
Finally, we employ a rule‐based interpreter to map the symbolic output back into natural language, yielding the final answer.

完整的prompt可以见appendix


## Experimental Setup

### Datasets
我们采用https://huggingface.co/datasets/shuyi-zsy/LLMSR/blob/main/llmsr/的数据集完成我们的工作。

### Model Architecture and Fine-tuning
我们使用Llama-3-8B-Instruct作为解析器和LLM Verifier的基础模型。
考虑到Llama-3-8B-Instruct对于Z3的语法的ground truth和推理能力一般，我们采用了o3-mini-high作为base model来生成Z3表达式。

We set the temperature as 0.2~0.3 in the Question Parser, 0.5~0.6 in the Combined and Cot Parser, 0.1~0.2 in the LLama Verifier.

we use the 24 samples in the datasets to fine-tune the base Llama-3-8B-Intruct with LoRA. We set the low-rank dimension as 32, the learning rate as 2e-5, training epochs as 6, batch size as 2. During inference, we set the temperature as 0 (i.e., greedy decoding) and the max sample length as 2,048. All our experiments can be conducted on 2×H20 GPU with 96GB of memory.

### Evaluation Metrics

- **Question_Macro_F1**
  We define Question_Macro_F1 as the macro‐averaged F1 score computed over the set of all atomic conditions that must be extracted from the input question. Each distinct condition constitutes its own class; true positives, false positives, and false negatives are counted per class, and then F1 is averaged uniformly across classes. This metric thus captures the model’s ability to recover every necessary constraint for downstream reasoning, regardless of class frequency.

- **Statement_Macro_F1**
  Statement_Macro_F1 denotes the macro‐averaged F1 score for segmenting and identifying individual reasoning statements and their associated evidence spans within the chain of thought. We treat each span type (statement vs. evidence) as a separate class and evaluate extraction quality via both lexical and semantic overlap against ground truth. Precision and recall are computed per class and averaged, ensuring balanced evaluation across all span categories.

- **Statement_Evidence_Macro_F1**
  Statement_Evidence_Macro_F1 measures the macro‐averaged F1 over pairwise links between extracted statements and their corresponding evidence. Each possible statement–evidence pairing is treated as a binary classification task (linked vs. unlinked). We compute class‐wise precision and recall for the “linked” label and average the resulting F1 scores across all statement–evidence candidates to assess the model’s ability to reconstruct the intended argumentative structure.

- **Reasoning_F1**
  Reasoning_F1 is the macro‐averaged F1 score for the final entailment verification between each correctly extracted statement–evidence pair. We frame logical deduction as a binary entailment decision (entails vs. does not entail). For all validated pairs, we compute precision and recall on the “entails” class and average the F1 scores uniformly, thereby evaluating end‐to‐end correctness of the tool‐augmented reasoning pipeline.


## Results and Conclusion
Table 1 presents the primary evaluation results for our base and fine-tuned models. The results include 4 performance metrics detailed in 4.x: Question_Macro_F1, Statement_Macro_F1, Statement_Evidence_Macro_F1 and Reasoning_F1. We have Four major findings.

| Approach | Question_F1 | Statement_F1 | Statement_Evidence_F1 | Reasoning_F1 |
|------|-------------|--------------|------------------------|--------------|
| Baseline | 0.7192 | |  |  |
| Base Question Parser | 0.7192 | |  |  |
| Base Combined Parser | 0.7087 | 0.4247 | 0.168 |  |
| Fine-tuned Combined Parser(Preprocessed by Llama3) | 0.5078 | 0.3979 | 0.1417 | |
| Llama Verifier |  | | | 0.067 |
| O3-mini-high Verifier |  |  |  | 0.124 |
| Z3 Verifier(72%) |  | | | 0.078 |

### Main Findings
1. 问题解析和推理的温度相对较低（确定性）0.2
2. 思维解析的温度不能太高也不能太低，0.5～0.6，不能太确定也不能太发散？？发散的话不助于对于evidence的提取难度更大，verification难度也更大。
3. 微调，思维链只需要较少的样本就能学习到，但是问题解析大幅下降，原因有待考量。更多的样本，才能提高
4. Z3 solver的成功率在72%的情况下Reasoning_F1为0.078，若解析100%成功的话Reasoning_F1的效果可以接近o3-mini-high的推理能力。






## References
@misc{zhang2024smalllanguagemodelsneed,
      title={Small Language Models Need Strong Verifiers to Self-Correct Reasoning}, 
      author={Yunxiang Zhang and Muhammad Khalifa and Lajanugen Logeswaran and Jaekyeom Kim and Moontae Lee and Honglak Lee and Lu Wang},
      year={2024},
      eprint={2404.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.17140}, 
}

@misc{paul2024refinerreasoningfeedbackintermediate,
      title={REFINER: Reasoning Feedback on Intermediate Representations}, 
      author={Debjit Paul and Mete Ismayilzada and Maxime Peyrard and Beatriz Borges and Antoine Bosselut and Robert West and Boi Faltings},
      year={2024},
      eprint={2304.01904},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2304.01904}, 
}

@misc{pan2023logiclmempoweringlargelanguage,
      title={Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning}, 
      author={Liangming Pan and Alon Albalak and Xinyi Wang and William Yang Wang},
      year={2023},
      eprint={2305.12295},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2305.12295}, 
}