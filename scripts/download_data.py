"""Download FiQA and print verified split sizes."""
import json

from finqa.data.fiqa import download, split_stats

if __name__ == "__main__":
    d = download()
    print(json.dumps(split_stats(d), indent=1))
