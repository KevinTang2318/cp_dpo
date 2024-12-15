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


## Analysis on train and val preference datasets

**aqua train statistics:**\
Correct answer matches ground truth: 51/177\
Wrong answer deviates from ground truth: 136/177

**aqua val statistics:**\
Correct answer matches ground truth: 14/51\
Wrong answer deviates from ground truth: 42/51

**strategy_qa train statistics:**\
Correct answer matches ground truth: 1005/1599\
Wrong answer deviates from ground truth: 1005/1599

**strategy_qa val statistics:**\
Correct answer matches ground truth: 302/457\
Wrong answer deviates from ground truth: 302/457

**coin_flip train statistics:**\
Correct answer matches ground truth: 160/350\
Wrong answer deviates from ground truth: 160/350

**coin_flip val statistics:**\
Correct answer matches ground truth: 46/100\
Wrong answer deviates from ground truth: 46/100

**object_tracking train statistics:**\
Correct answer matches ground truth: 13/525\
Wrong answer deviates from ground truth: 508/525

**object_tracking val statistics:**\
Correct answer matches ground truth: 4/150\
Wrong answer deviates from ground truth: 144/150

**last_letter train statistics:**\
Correct answer matches ground truth: 1/350\
Wrong answer deviates from ground truth: 350/350

**last_letter val statistics:**\
Correct answer matches ground truth: 1/100\
Wrong answer deviates from ground truth: 100/100

**bigbench_date train statistics:**\
Correct answer matches ground truth: 130/258\
Wrong answer deviates from ground truth: 237/258

**bigbench_date val statistics:**\
Correct answer matches ground truth: 33/74\
Wrong answer deviates from ground truth: 72/74
