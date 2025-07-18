import pickle

with open("pkls/raw_data/jam-cgpt-raw170k.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Type of data: {type(data)}")
print(f"Length of data: {len(data)}")
for i, example in enumerate(data[:5]):
    print(f"Example {i}: {example}")
