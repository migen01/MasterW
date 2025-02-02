from datasets import load_dataset
squad_dataset = load_dataset('squad', split='validation')
#print(squad_dataset[0:2])  # Print the first entry

print(squad_dataset.train_test_split(test_size=0.2))
# For a different dataset format:
