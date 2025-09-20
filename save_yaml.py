import pickle
def save_yaml_to_file(yaml_text,filename):
    with open(filename, "wb") as f:
        pickle.dump(yaml_text, f)
    print(f"✅ YAML saved to {filename}")
