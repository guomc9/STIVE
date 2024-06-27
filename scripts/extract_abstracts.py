import csv
import argparse
from collections import Counter
import json

def read_abstract_prompts(file_path):
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        abstracts = [row['Abstracts Prompt'] for row in reader]
    return abstracts

def extract_dollar_words(abstracts):
    dollar_words = []
    for abstract in abstracts:
        words = abstract.split()
        for word in words:
            if word.startswith('$'):
                if word.endswith('.') or word.endswith(','):
                    word = word[:-1]
                dollar_words.append(word[1:])
    return dollar_words


def categorize_words(words):
    word_count = Counter(words)
    global_abstracts = [word for word, count in word_count.items() if count > 1]
    private_abstracts = [word for word, count in word_count.items() if count == 1]
    return global_abstracts, private_abstracts

def main():
    parser = argparse.ArgumentParser(description='Analyze abstract prompts from a CSV file.')
    parser.add_argument('-f', '--file_path', type=str, help='Path to the CSV file')
    parser.add_argument('-o', '--output_path', type=str, help='Path to the output JSON file')
    args = parser.parse_args()

    abstracts = read_abstract_prompts(args.file_path)
    dollar_words = extract_dollar_words(abstracts)
    global_abstracts, private_abstracts = categorize_words(dollar_words)

    print("Global Abstracts:", global_abstracts)
    print("Private Abstracts:", private_abstracts)
    
    result = {
        "abstracts": global_abstracts + private_abstracts, 
        "global_abstracts": global_abstracts,
        "private_abstracts": private_abstracts
    }

    with open(args.output_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(result, jsonfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()