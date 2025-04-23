# ReadME
20250416 陈丹纯（dcchen）

# 写在前面
首先非常感谢主办方给我补交代码的机会。
其次，提交的所有代码都是在Test A阶段结束前完成的。不过因为Test A阶段我改了很多次代码、尝试了很多种方法（并且互相组合）、prompt和参数，而且每次运行的时候是几个脚本（比如qp，verification）依次执行，把所有代码整合起来的过程中可能不小心弄错了参数、prompt等，而且我的eval.py脚本不知为何无法执行，无法再次验证自己的results.json的答案。所以不能保证这份代码运行出来的结果完全与Test A阶段的某一次结果完全一模一样。预计结果在#261991、#262709、#257571之间，其中#261991的reasoning score不具备参考性，因为这次提交结果忘记verification了。
为没有提前整合并保存好代码给你们造成的评估困扰而深表歉意！代码运行/Test A评估有任何问题请随时联系我，谢谢！

# env
```shell
pip install -r requirements.txt
```
verification阶段采用了azure openai/o3-mini作为推理模型，其他阶段都是采用未经finetune的Llama-3-8B-Instruct。
o3-mini在`.env`中设置：
```
AZURE_OPENAI_ENDPOINT=xxx
OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_API_KEY=xxx
```


# How to run

请直接下载meta或者model-scope的llama-3-8B-instruct保存到本地路径中，eg:`/datacenter/models/LLM-Research/Llama-3-8B-Instruct`。
将待测试数据文件放入`data/`中。

```shell
python run_pipeline.py --input data/Public_Test_A.json --output_dir results/ --model_path model_path 模型本地路径 
```
 
因为运行时间很久，一个多小时，建议挂起：
```shell
nohup python run_pipeline.py --input data/Public_Test_A.json --output_dir results/ --model_path 模型本地路径 > ./logs/run.log 2>&1
```
运行结束之后会在`results/`中看到四个文件，其中`results/results_*.json`就是最终的结果。


如果没有o3-mini，可以注释掉`run_pipeline.py`中步骤四，最终结果看`results/step3_*.json`我在步骤三中提前用llama3推理了，但是o3-mini推理效果更好。
```python
 # 步骤4: 运行inference_gpt.py
    logging.info("开始步骤4: 运行inference_gpt.py")
    cmd3 = f"python inference_gpt.py --input {step3_output} --output {final_output} --model {args.verify_model}"
    if not run_command(cmd3):
        logging.error("步骤4失败，终止流水线")
        return
```


