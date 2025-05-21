from datasets import load_dataset
squad_dataset = load_dataset('squad', split='validation')
squad_dataset_v2 = load_dataset('squad_v2',split='validation')


raw_top_entries = squad_dataset_v2.select(range(220))
top_5_entries = raw_top_entries.filter(lambda example: example.get('answers', {}).get('text', []))
print(top_5_entries)