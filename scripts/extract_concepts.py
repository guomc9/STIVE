import csv
import argparse
from collections import Counter
import json

def read_concept_prompts(file_path):
    video_concepts = {}
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_name = row['Video name']
            concept_prompt = row['Concepts Prompt']
            if video_name not in video_concepts:
                video_concepts[video_name] = []
            video_concepts[video_name].append(concept_prompt)
    return video_concepts

def extract_dollar_words(concept):
    dollar_words = []
    words = concept.split()
    for word in words:
        if word.startswith('$'):
            if word.endswith('.') or word.endswith(','):
                word = word[:-1]
            dollar_words.append(word)
    return dollar_words

def categorize_words(words):
    word_count = Counter(words)
    global_concepts = [word for word, count in word_count.items() if count > 1]
    private_concepts = [word for word, count in word_count.items() if count == 1]
    return global_concepts, private_concepts

def main():
    parser = argparse.ArgumentParser(description='Analyze concept prompts from a CSV file.')
    parser.add_argument('-f', '--file_path', type=str, help='Path to the CSV file', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='Path to the output JSON file', required=True)
    args = parser.parse_args()

    video_concepts = read_concept_prompts(args.file_path)

    result = {
        "video_concepts": {},
        "concepts": [],
        "global_concepts": [],
        "private_concepts": []
    }

    all_dollar_words = []

    for video_name, concepts in video_concepts.items():
        dollar_words = []
        for concept in concepts:
            words = extract_dollar_words(concept)
            dollar_words.extend(words)
            all_dollar_words.extend(words)
        result["video_concepts"][video_name] = dollar_words

    global_concepts, private_concepts = categorize_words(all_dollar_words)
    result["concepts"] = all_dollar_words
    result["global_concepts"] = global_concepts
    result["private_concepts"] = private_concepts

    with open(args.output_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(result, jsonfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()