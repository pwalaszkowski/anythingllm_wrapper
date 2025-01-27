import os
import evaluate
import csv

from datetime import datetime
from helpers import load_pdf_text

def rouge_calculation(prediction_file):
    predictions = load_pdf_text(prediction_file)
    references = load_pdf_text('rouge_output/reference.pdf')

    min_len = min(len(predictions), len(references))
    predictions = predictions[:min_len]
    references = references[:min_len]

    references = [[ref] for ref in references]

    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)

    print('ROUGE-1 Score:', results['rouge1'])
    print('ROUGE-2 Score:', results['rouge2'])
    print('ROUGE-L Score:', results['rougeL'])
    print('ROUGE-LSum Score:', results['rougeLsum'])

    # Generate timestamp and output file name
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f'{timestamp}_rouge_results.csv'

    with open(os.path.join('rouge_output', file_name), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'F1-score'])
        writer.writerow(['ROUGE-1', results['rouge1']])
        writer.writerow(['ROUGE-2', results['rouge2']])
        writer.writerow(['ROUGE-L', results['rougeL']])
        writer.writerow(['ROUGE-LSum', results['rougeLsum']])
