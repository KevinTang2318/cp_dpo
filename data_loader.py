import json

def load_aqua_data(file_path):
    with open(file_path, "r") as data_file:
        samples = data_file.readlines()

    aqua_dataset = []
    for sample in samples:
        aqua_dataset.append(json.loads(sample))

    return aqua_dataset


def load_strategy_qa_data(file_path):
    with open(file_path, "r") as data_file:
        dataset = json.loads(data_file.read())

    return dataset["examples"]

def load_coin_flip_data(file_path):
    with open(file_path, "r") as data_file:
        dataset = json.loads(data_file.read())

    return dataset["examples"]

def load_object_tracking_data(file_path):
    with open(file_path, "r") as data_file:
        dataset = json.loads(data_file.read())

    return dataset["examples"]

def load_last_letter_data(file_path):
    with open(file_path, "r") as data_file:
        dataset = json.loads(data_file.read())

    return dataset["examples"]


def load_bigbench_date_data(file_path):
    with open(file_path, "r") as data_file:
        dataset = json.loads(data_file.read())

    return dataset["examples"]