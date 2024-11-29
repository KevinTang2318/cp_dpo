# cp_dpo


## Test Results

The following table shows the accuracy scores of all experiments conducted in this project, regarding the four datasets we selected. All tests were conducted using the Llama3 8B instruct model, either a base version with Constrastive Prompting (CP) or a version fine-tuned using Constrastive Prompting and Direct Preference Optimization.

|              | AQuA         | StragegyQA     | CoinFlip     | BigBench Object Tracking   |
|--------------|--------------|----------------|--------------|----------------------------|
| Zero-Shot CP (baseline) | 0.2874015748031496 | 0.6336980306345733 | 0.462 | 0.31066666666666665 |