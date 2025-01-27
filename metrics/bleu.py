import os
import evaluate
import csv

from datetime import datetime
from helpers import load_pdf_text


def bleu_calculation(prediction_file):
    predictions = load_pdf_text(prediction_file)
    references = load_pdf_text('bleu_output/reference.pdf')

    min_len = min(len(predictions), len(references))
    predictions = predictions[:min_len]
    references = references[:min_len]

    references = [[ref] for ref in references]

    bleu = evaluate.load('bleu')
    results = bleu.compute(predictions=predictions, references=references)

    print('BLEU Score:', results['bleu'])
    print('Precisions:', results['precisions'])
    print('Brevity Penalty:', results['brevity_penalty'])
    print('Length Ratio:', results['length_ratio'])
    print('Translation Length:', results['translation_length'])
    print('Reference Length:', results['reference_length'])

    # Generate timestamp and output file name
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f'{timestamp}_bleu_results.csv'

    with open(os.path.join('bleu_output', file_name), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Score'])
        writer.writerow(['BLEU', results['bleu']])
        writer.writerow(['Precisions (n-gram 1-4)', results['precisions']])
        writer.writerow(['Brevity Penalty', results['brevity_penalty']])
        writer.writerow(['Length Ratio', results['length_ratio']])
        writer.writerow(['Translation Length', results['translation_length']])
        writer.writerow(['Reference Length', results['reference_length']])
