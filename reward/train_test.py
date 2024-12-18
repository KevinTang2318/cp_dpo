import json
import random

# File paths
input_file = "data.jsonl"  # Replace with your input file path
training_file = "training.jsonl"
validation_file = "validation.jsonl"
testing_file = "testing.jsonl"

# Split ratio for training and testing
split_ratio = 0.7  # 80% for training, 20% for testing
val_ratio = 0.1

# Read the JSONL file
with open(input_file, "r") as f:
    lines = f.readlines()

# Shuffle the data
random.shuffle(lines)

# Split the data
split_train_index = int(len(lines) * split_ratio)
split_val_index = int(len(lines) * (split_ratio + val_ratio))
training_data = lines[:split_train_index]
validation_data = lines[split_train_index:split_val_index]
testing_data = lines[split_val_index:]

# Write training data
with open(training_file, "w") as f:
    f.writelines(training_data)
with open(validation_file, "w") as f:
    f.writelines(validation_data)

# Write testing data
with open(testing_file, "w") as f:
    f.writelines(testing_data)

print(f"Data split complete: {len(training_data)} training samples, {len(validation_data)} validation samples, {len(testing_data)} testing samples.")
