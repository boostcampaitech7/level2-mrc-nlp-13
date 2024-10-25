import json
import json
import difflib
import sys

from datasets import load_from_disk

# Load datasets
datasets = load_from_disk("../../data/test_dataset")
validation_dataset = datasets["validation"]

# Create a mapping from id to question
id_to_question = {item['id']: item['question'] for item in validation_dataset}

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def compare_json(file1, file2):
    # Load JSON files
    json1 = load_json(file1)
    json2 = load_json(file2)

    # Convert JSON to strings (sorted by keys to ignore order differences)
    str1 = json.dumps(json1, sort_keys=True, indent=2, ensure_ascii=False).splitlines()
    str2 = json.dumps(json2, sort_keys=True, indent=2, ensure_ascii=False).splitlines()

    # Calculate differences
    differ = difflib.Differ()
    diff = list(differ.compare(str1, str2))

    # Track the last processed ID to avoid repeating questions
    last_id = None

    # Output differences with corresponding questions
    for line in diff:
        if line.startswith('- ') or line.startswith('+ '):
            # Extract ID from the line if possible
            try:
                # Extract ID from the line assuming format is "id": "answer"
                id_start = line.find('"') + 1
                id_end = line.find('"', id_start)
                id_value = line[id_start:id_end]

                # Only print question if it's a new ID being processed
                if id_value != last_id:
                    question = id_to_question.get(id_value, "Unknown Question")
                    print(f'\033[97mQuestion: {question}\033[0m')
                    last_id = id_value

            except Exception as e:
                print(f'\033[97mError retrieving question\033[0m')

            # Print the difference with color coding
            color_code = '\033[91m' if line.startswith('- ') else '\033[92m'
            print(f'{color_code}    {line}\033[0m')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <file1.json> <file2.json>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    compare_json(file1, file2)