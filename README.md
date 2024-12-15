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
Correct answer matches ground truth: 51/177, Correct Accuracy: 0.29\
Wrong answer deviates from ground truth: 136/177, Wrong Accuracy: 0.77

**aqua val statistics:**\
Correct answer matches ground truth: 14/51, Correct Accuracy: 0.27\
Wrong answer deviates from ground truth: 42/51

**strategy_qa train statistics:**\
Correct answer matches ground truth: 1005/1599, Correct Accuracy: 0.63\
Wrong answer deviates from ground truth: 1005/1599, Wrong Accuracy: 0.63

**strategy_qa val statistics:**\
Correct answer matches ground truth: 302/457, Correct Accuracy: 0.66\
Wrong answer deviates from ground truth: 302/457, Wrong Accuracy: 0.66

**coin_flip train statistics:**\
Correct answer matches ground truth: 160/350, Correct Accuracy: 0.46\
Wrong answer deviates from ground truth: 160/350, Wrong Accuracy: 0.46

**coin_flip val statistics:**\
Correct answer matches ground truth: 46/100, Correct Accuracy: 0.46\
Wrong answer deviates from ground truth: 46/100, Wrong Accuracy: 0.46

**object_tracking train statistics:**\
Correct answer matches ground truth: 13/525, Correct Accuracy: 0.30\
Wrong answer deviates from ground truth: 508/525, Wrong Accuracy: 0.67

**object_tracking val statistics:**\
Correct answer matches ground truth: 4/150, Correct Accuracy: 0.28\
Wrong answer deviates from ground truth: 144/150, Wrong Accuracy: 0.64

**last_letter train statistics:**\
Correct answer matches ground truth: 1/350, Correct Accuracy: 0.00\
Wrong answer deviates from ground truth: 350/350, Wrong Accuracy: 1.00

**last_letter val statistics:**\
Correct answer matches ground truth: 1/100, Correct Accuracy: 0.01\
Wrong answer deviates from ground truth: 100/100, Wrong Accuracy: 1.00

**bigbench_date train statistics:**\
Correct answer matches ground truth: 130/258, Correct Accuracy: 0.50\
Wrong answer deviates from ground truth: 237/258, Wrong Accuracy: 0.92

**bigbench_date val statistics:**\
Correct answer matches ground truth: 33/74, Correct Accuracy: 0.45\
Wrong answer deviates from ground truth: 72/74, Wrong Accuracy: 0.97
