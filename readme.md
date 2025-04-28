# env
```shell
pip install -r requirements.txt
```

API for Azure/openai are required if you wanna use the openai LLM. You could set at `.env`:
```
AZURE_OPENAI_ENDPOINT=xxx
OPENAI_API_VERSION=2025-03-01-preview
AZURE_OPENAI_API_KEY=xxx
```



# How to run
The code has integrated all the modules in pipeline. 
To run the code, download Llama-3-8B-instruct to local path as `model_path` or use azure/openai model_name (e.g. o3-mini-high).
Place your data file in `data/`. After running the pipeline, the output of middle modules can be seen in `results/`.

## Run the Best Combination
```shell
python -m src.best_pipeline --input data/Public_Test_A.json --output results/Test_A_results.json --model [llama_model_path]
```

## Run Customized Pipeline
Set parser as 'icl' to run base model or 'ft' to run finetuned model.
Set verifier as 'llm' to run LLM Verifier or 'z3' to run Z3-Augmented Verifier.
`--model` could be set as model_path of Llama or  model_name of azure/openai (e.g. o3-mini-high).
model_params should be set as dict. e.g. 'temperature=0.2,top_p=0.8' or 'reasoning_effort=high'. 

example:
```shell
python -m src.llmsr_pipeline --input data/Public_Test_A.json --output results/Test_A_results.json --combined --combined_parser icl --combined_model  [llama_model_path] --combined_params 'temperature=0.5'  --verify --verifier llm --verifier_model [openai_model_name] --verifier_params 'reasoning_effort=low' 
```

```shell
python -m src.llmsr_pipeline --input results/Test_A_results_final_20250426_064748.json --output results/Test_A_results.json --qp --qp_parser icl --qp_model '/datacenter/models/LLM-Research/Llama-3-8B-Instruct' --qp_params 'temperature=0.3,do_sample=False,top_p=0.8' --verifier z3 --verifier_model o3-mini-high --verifier_params 'reasoning_effort=high' 
```


# Fine-tune
```shell
python -m src.parsers.finetune --do_train --reference_file data/Final_Selection_Train_v2.json --preprocessed_file Train_o3-mini-high_results.json --output_dir 'path/to/save/finetuned/model' --model 'path/to/llama'
```
