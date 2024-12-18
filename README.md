# Enhancing Large Language Models’ Reasoning Capabilities through Contrastive Prompting (CP) and Direct Preference Optimization (DPO)

**Authors: Altria Wang (zw2958), Kevin Tang (kt2942)**

In this project, we explore the effect of using Contrastive Prompting (CP) and Direct Preference Optimization to enhance the reasoning capability of Llama3-8B-instruct model. Detailed experimental setup is specified in our final project report, which is also included in this repository.

This repository contains the data and code to generate preference datasets and perform data preprocessing. All raw data were obtained from this [repo](https://github.com/yao8839836/cp) (original study on Contrastive Prompting).

## Repository Structure
- **datasets**: contains the original data for the six selected reasoning tasks
- **data**: contains the train, val, and test splits for all datasets
- **llm_output**: contains the datasets combined with generated response from Llama3-8B-instruct (CP)
- **preference_data**: contains the processed preference datasets
- ***.py**: functional codes for data preprocessing and LLM response generation

## Computational Resource Requirements
We recommend running the code in this repo on a GCP VM that has the following configurations:

- **Region**: us-east1
- **Device Type**: G2 with 1 L4 GPU
- **Persistent Disk Size**: 200GB
- **OS**: Deep Learning VM for PyTorch 2.4 with CUDA 12.4 M126

You would also need to install the `transformers` library and all its dependency on the VM to successfully run inference for Llama3.

## Execution Instructions for data generation and analysis
1. Clone this repo onto your VM
2. Configure an environment variable called `HF_TOKEN` with your huggingface token ad the value.
3. Run `data_preprocessing.py` on all datasets to generates the datasets with LLM generated responses.
    ```
    python3 data_preprocessing.py aqua strategy_qa coin_flip object_tracking last_letter bigbench_date
    ```
    You can also select specific dataset(s) to process by only specifying its name.
4. Run `generate_preference_dataset.py` on all datasets to generate the preference datasets which will be used for DPO fine-tuning:
    ```
    python3 generate_preference_datasets.py
    ```
    To run this code for specific datasets, comment out the function invocation for other datasets in the main section of this code.
5. Run `calculate_accuracy.py` to get the test accuracies on all six reasoning tasks using Zero-Shot prompting:
    ```
    python3 calculate_accuracy.py aqua strategy_qa coin_flip object_tracking last_letter bigbench_date
    ```
    You can also select specific dataset(s) to process by only specifying its name.
6. Run `analysis.py` to calculate the precision for the generated responses on train and val sets using CP:
    ```
    python3 analysis.py all
    ```
    You can also select specific dataset(s) to process by only specifying its name.

Finally, you would need to use all data under `preference_data` in the DPO stage to fine-tune the model.

# Execution Instructions for DPO training and output parsing 
1. Training the model: 
    Run the following code with your desired configurations. The following one is the paramter setting we used.
    ```
    CUDA_VISIBLE_DEVICES=0,1 python dpo_training.py \
    --model_type auto \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --train_file_dir ./data/reward/train \
    --validation_file_dir ./data/reward/validation \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 250 \
    --max_eval_samples 10 \
    --max_steps 100 \
    --eval_steps 20 \
    --save_steps 50 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --output_dir outputs-dpo-llama-v10 \
    --target_modules "k_proj, q_proj, v_proj" \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --torch_dtype float16 \
    --fp16 True \
    --device_map auto \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --cache_dir ./cache
    ``` 
2. Merge the LoRA with the original model and modify the directories as desired:
    ```
    python merge_peft_adapter.py --model_type llama \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct --lora_model outputs-dpo-llama-v10 --output_dir merged-dpo-v10/
    ```
3. Run the inference code and modify the directory (of the output file) in the code at line 258 
    ```
    python inference.py --model_type llama --base_model merged-dpo-v10
    ```
4. Parse the output by changing the json file of the output in line 4, then run:
    ```
    python extract_answer.py
    ```
    This should give you the accuracy score. 

Credit to GitHub MedicalGPT (@author:XuMing(xuming624@qq.com)) for step 1, 2, 3. 
## Results

### Tests statistics for CP + DPO fine-tuning

The following table shows the accuracy scores of all experiments conducted in this project, regarding the six datasets we selected. All tests were conducted using the Llama3-8B-instruct model. We calculated accuracies of responses generated by the standard Llama3-8B-instruct model and the model fine-tuned with CP+DPO.


|              | Zero-Shot (baseline)    | Zero-Shot-CP-DPO |
|--------------|----------------------------|-----------------|
| AQuA | 0.39 | 0.38 |
| Strategy-QA | 0.66 | 0.63 |
| Coin Flip | 0.46 | 0.46 |
| Object Tracking | 0.29| 0.02 |
| Last Letters | 0.14 | 0.38 |
| BigBench Date | 0.43 | 0.27 |


We compare the difference in the baseline Zero-shot prompting and Zero-shot result after training with DPO method from CP method. Unfortunately, we did not find much significance in the performance. The performance pairs are mostly very similar to each other except for the Last Letter pair. In the Last Letter finetuning, both preferred answer and not preferred answer are purely letter combinations (such as "aabb"). This means that model face great difficulty in inferring the intrinsic pattern of the letters from the answers. 


### Analysis on train and val preference datasets

The following tables demonstrates the precision for correct and wrong answer generated using Contrastive Prompting. We collected these data to investigate how accurate CP is and what kind of impact it has on the final fine-runed model.

**Precision on generated answers in training sets**

|              | Correct Precision   | Wrong Precision |
|--------------|---------------------|-----------------|
| AQuA | 0.29 | 0.77 |
| Strategy-QA | 0.63 | 0.63 |
| Coin Flip | 0.46 | 0.46 |
| Object Tracking | 0.01 | 0.99 |
| Last Letters | 0.50 | 0.92 |
| BigBench Date | 0.30 | 0.67 |

**Precision on generated answers in validation sets**

|              | Correct Precision   | Wrong Precision |
|--------------|---------------------|-----------------|
| AQuA | 0.27 | 0.82 |
| Strategy-QA | 0.66 | 0.66 |
| Coin Flip | 0.46 | 0.46 |
| Object Tracking | 0.01 | 1.00 |
| Last Letters | 0.45 | 0.97 |
| BigBench Date | 0.28 | 0.64 |

We can derive several insights from the precision compar-
isons. First, the precision for the generated correct answers
is relatively low across all tasks, with the highest score
of 0.63 observed on the Strategy QA training set (assume
we only compare training sets here). Notably, the model
performed extremely poorly on the Last Letter task, achieving
a correct precision close to zero, which indicates that very
few questions were interpreted or solved correctly. Addition-
ally, the model demonstrated limited capabilities in solving
mathematical problems. All such undesired behaviors likely
due to the constrained model size of LLaMA3-8B-Instruct,
since the experiment result shown by Yan et al. are relatively
higher when using GPT-4.

The relatively low precision on correct responses suggests
that fine-tuning with such data may mislead the model by rein-
forcing suboptimal patterns, potentially degrading its overall
performance. However, a key observation is the consistency
between Table II (training precision) and Table III (validation
precision), which indicates minimal distributional differences
between the training and validation datasets. This stability
could contribute to improved generalization and better per-
formance of the fine-tuned model.
