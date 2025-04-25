# LLMSR Pipeline System

这个目录包含了LLMSR项目的Pipeline系统，用于构建和运行灵活的解析和验证流程。

## 文件结构

- `node.py`: Pipeline节点基类
- `parser_node.py`: 解析器节点类
- `verifier_node.py`: 验证器节点类
- `pipeline_builder.py`: Pipeline构建器类
- `pipeline_runner.py`: Pipeline运行器类

## 使用方法

### 命令行使用

可以通过命令行参数来配置和运行Pipeline：

```bash
# 使用命令行参数
python -m src.run_pipeline \
    --input data/test.json \
    --output results/results_full.json \
    --combined \
    --combined_parser icl \
    --combined_model gpt-4 \
    --combined_params "temperature=0.8,top_p=0.95,max_tokens=2048" \
    --qp \
    --qp_parser icl \
    --qp_model gpt-4 \
    --verify \
    --verifier llm \
    --verifier_model gpt-4 \
    --verifier_params "temperature=0.5,top_p=0.9,max_tokens=1024" \
    --batch_size 10 \
    --max_retries 3

# 使用配置文件
python -m src.run_pipeline \
    --input data/test.json \
    --output results/results_full.json \
    --config configs/default_pipeline.json
```

### 代码中使用

也可以在代码中使用Pipeline系统：

```python
from src.pipeline.pipeline_builder import PipelineBuilder
from src.pipeline.pipeline_runner import PipelineRunner

# 创建 Pipeline 构建器
builder = PipelineBuilder("LLMSR_Pipeline")

# 添加组合解析节点
builder.add_parser(
    name="combined_parser",
    parser_type="icl",
    model_name="gpt-4",
    task_type="combined",
    model_params={"temperature": 0.8, "top_p": 0.95, "max_tokens": 2048},
    output_file="results/results_combined.json"
)

# 添加问题解析节点
builder.add_parser(
    name="qp_parser",
    parser_type="icl",
    model_name="gpt-4",
    task_type="qp",
    model_params={"temperature": 0.7, "top_p": 0.9, "max_tokens": 1024},
    output_file="results/results_qp.json"
)

# 添加验证节点
builder.add_verifier(
    name="verifier",
    verifier_type="llm",
    model_name="gpt-4",
    model_params={"temperature": 0.5, "top_p": 0.9, "max_tokens": 1024},
    output_file="results/results_verified.json"
)

# 构建 Pipeline
pipeline = builder.build()

# 运行 Pipeline
results = PipelineRunner.run(
    pipeline=pipeline,
    input_file="data/test.json",
    output_file="results/results_final.json",
    batch_size=10,
    max_retries=3
)
```

## 配置文件格式

配置文件是一个JSON文件，包含以下字段：

```json
{
    "combined": true,
    "combined_parser": "icl",
    "combined_model": "gpt-4",
    "combined_params": {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 2048
    },
    
    "qp": true,
    "qp_parser": "icl",
    "qp_model": "gpt-4",
    "qp_params": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1024
    },
    
    "cp": false,
    "cp_parser": "icl",
    "cp_model": "gpt-4",
    "cp_params": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1024
    },
    
    "verify": true,
    "verifier": "llm",
    "verifier_model": "gpt-4",
    "verifier_params": {
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 1024
    },
    
    "batch_size": 10,
    "max_retries": 3
}
```

## 自定义节点

可以通过继承`Node`类来创建自定义节点：

```python
from src.pipeline.node import Node

class CustomNode(Node):
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.kwargs = kwargs
    
    def process(self, data, **kwargs):
        # 处理数据
        # ...
        return processed_data
```

然后可以将自定义节点添加到Pipeline中：

```python
custom_node = CustomNode("custom_node", param1="value1", param2="value2")
pipeline.add_node(custom_node)
```

## 节点顺序

Pipeline中的节点按照添加的顺序依次执行。每个节点的输出将作为下一个节点的输入。

## 错误处理

Pipeline系统会记录每个节点的运行日志，可以在`logs`目录下查看。如果某个节点运行失败，Pipeline会抛出异常并停止执行。
