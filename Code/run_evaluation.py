import json
from datasets import load_dataset
from tqdm import tqdm
from vlm_engine import VLMEngine
from metrics_utils import clean_json_string, compute_f1_score

def main():
    print("Loading CORD Dataset for Evaluation...")
    dataset = load_dataset("naver-clova-ix/cord-v2")
    test_subset = dataset['test'].select(range(10)) 

    # Initialize Model
    engine = VLMEngine()

    total_acc, total_f1, valid_docs = 0, 0, 0

    print("\nRunning Zero-Shot Extraction Pipeline...")
    for i, example in enumerate(tqdm(test_subset)):
        
        raw_gt = json.loads(example['ground_truth'])
        clean_gt = raw_gt.get('gt_parse', {})
        image = example['image'].convert('RGB')
        
        raw_output = engine.analyze_document(image, task_type="extraction")
        pred_json = clean_json_string(raw_output)

        if not clean_gt:
            continue

        valid_docs += 1
        acc, f1 = compute_f1_score(clean_gt, pred_json)
        total_acc += acc
        total_f1 += f1

        if i < 3:
            print(f"\n--- Document {i+1} ---")
            print(f"Model RAW Output: {raw_output}")
            print(f"F1 Score: {f1:.4f}")

    if valid_docs > 0:
        print("\n" + "="*40)
        print("      ZERO-SHOT EVALUATION METRICS")
        print("="*40)
        print(f"Total Documents    : {valid_docs}")
        print(f"Average Accuracy   : {(total_acc / valid_docs) * 100:.2f}%")
        print(f"Average F1-Score   : {(total_f1 / valid_docs):.4f}")
        print("="*40)

if __name__ == "__main__":
    main()
