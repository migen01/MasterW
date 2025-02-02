from ctypes import util
from os import fsync
import time
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

from datasets import load_dataset
squad_dataset = load_dataset('squad', split='validation')[0]

#top_5_entries = squad_dataset.select(range(5))

print(squad_dataset)