import json
import re

def clean_json_string(output_str):
    """Strips markdown and uses regex fallback for VLM syntax errors."""
    cleaned = re.sub(r"```json\s*", "", output_str)
    cleaned = re.sub(r"\s*```", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        rescued_dict = {}
        pattern = r'"([^"]+)"\s*:\s*(?:"([^"]*)"|([A-Za-z0-9\.,]+))'
        for match in re.finditer(pattern, cleaned):
            key = match.group(1)
            value = match.group(2) if match.group(2) is not None else match.group(3)
            rescued_dict[key] = value
        return rescued_dict

def extract_all_kv_pairs(d):
    """Flattens dictionary for strict key-value scoring."""
    items = set()
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                items.update(extract_all_kv_pairs(v))
            elif str(v).strip():
                items.add((str(k).strip().lower(), str(v).strip().lower()))
    elif isinstance(d, list):
        for item in d:
            items.update(extract_all_kv_pairs(item))
    return items

def compute_f1_score(true_dict, pred_dict):
    """Calculates accuracy and F1 score using flat tuple set intersection."""
    true_pairs = extract_all_kv_pairs(true_dict)
    pred_pairs = extract_all_kv_pairs(pred_dict)

    true_positives = len(true_pairs.intersection(pred_pairs))
    precision = true_positives / len(pred_pairs) if len(pred_pairs) > 0 else 0.0
    recall = true_positives / len(true_pairs) if len(true_pairs) > 0 else 0.0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = true_positives / len(true_pairs.union(pred_pairs)) if len(true_pairs.union(pred_pairs)) > 0 else 0.0
    
    return acc, f1
