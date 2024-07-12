import csv
import argparse
from collections import Counter
import json

def read_abstract_prompts(file_path):
    video_abstracts = {}
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_name = row['Video name']
            abstract_prompt = row['Abstracts Prompt']
            if video_name not in video_abstracts:
                video_abstracts[video_name] = []
            video_abstracts[video_name].append(abstract_prompt)
    return video_abstracts

def extract_dollar_words(abstract):
    dollar_words = []
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
    parser.add_argument('-f', '--file_path', type=str, help='Path to the CSV file', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='Path to the output JSON file', required=True)
    args = parser.parse_args()

    video_abstracts = read_abstract_prompts(args.file_path)

    result = {
        "video_abstracts": {},
        "abstracts": [],
        "global_abstracts": [],
        "private_abstracts": []
    }

    all_dollar_words = []

    for video_name, abstracts in video_abstracts.items():
        dollar_words = []
        for abstract in abstracts:
            words = extract_dollar_words(abstract)
            dollar_words.extend(words)
            all_dollar_words.extend(words)
        result["video_abstracts"][video_name] = dollar_words

    global_abstracts, private_abstracts = categorize_words(all_dollar_words)
    result["abstracts"] = all_dollar_words
    result["global_abstracts"] = global_abstracts
    result["private_abstracts"] = private_abstracts

    with open(args.output_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(result, jsonfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()