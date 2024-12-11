# cp_dpo


## Test Results

The following table shows the accuracy scores of all experiments conducted in this project, regarding the four datasets we selected. All tests were conducted using the Llama3 8B instruct model, either a base version with Constrastive Prompting (CP) or a version fine-tuned using Constrastive Prompting and Direct Preference Optimization.

|              | AQuA         | StragegyQA     | CoinFlip     | BigBench Object Tracking   |
|--------------|--------------|----------------|--------------|----------------------------|
| Zero-Shot CP (baseline) | 0.2545454545454545 | 0.6554054054054054 | 0.4351851851851852 | 0.3576158940397351 |