# cp_dpo


## Test Results

The following table shows the accuracy scores of all experiments conducted in this project, regarding the four datasets we selected. All tests were conducted using the Llama3 8B instruct model, either a base version with Constrastive Prompting (CP) or a version fine-tuned using Constrastive Prompting and Direct Preference Optimization.


|              | Zero-Shot CP (baseline)    |
|--------------|----------------------------|
| AQuA | 0.39 |
| Strategy-QA | 0.66 |
| Coin Flip | 0.46 |
| Object Tracking | 0.29|
| Last Letters | 0.14 |
| BigBench Date | 0.43 |