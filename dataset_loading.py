from datasets import load_dataset_builder, get_dataset_split_names, load_dataset

ds_builder = load_dataset_builder("vibhorag101/phr-mental-therapy-dataset-conversational-format-1024-tokens")

print(ds_builder.info.description)
print(ds_builder.info.features)
print(get_dataset_split_names("vibhorag101/phr-mental-therapy-dataset-conversational-format-1024-tokens"))
dataset = load_dataset("vibhorag101/phr-mental-therapy-dataset-conversational-format-1024-tokens", split="train")
print(dataset)