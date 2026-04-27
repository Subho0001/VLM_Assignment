import json
from document_processor import process_pdf_to_images
from vlm_engine import VLMEngine
from metrics_utils import clean_json_string

def main():
    # Path to your local sample PDF
    pdf_path = "wordpress-pdf-invoice-plugin-sample.pdf"

    # 1. Image Conversion
    pages = process_pdf_to_images(pdf_path)

    # 2. Initialize Model
    engine = VLMEngine()
    tasks = ["extraction", "signature", "form_fields"]
    all_results = {}

    print(f"\nStarting End-to-End Analysis for: {pdf_path}")
    
    for i, page_image in enumerate(pages):
        page_key = f"page_{i+1}"
        all_results[page_key] = {}

        print(f"\n" + "="*30)
        print(f" ANALYZING PAGE {i+1}")
        print("="*30)

        for task in tasks:
            print(f"Running Task: [{task}]...")
            raw_result = engine.analyze_document(page_image, task_type=task)

            # Route parsing based on output type
            if task in ["extraction", "form_fields"]:
                parsed_result = clean_json_string(raw_result)
                all_results[page_key][task] = parsed_result
            else:
                all_results[page_key][task] = raw_result

            print(f"  Result: {all_results[page_key][task]}")

    print("\n" + "#"*50)
    print(" FINAL STRUCTURED OUTPUT (Module 3 Deliverable)")
    print("#"*50)
    print(json.dumps(all_results, indent=2))

    # Save to disk
    output_filename = "extraction_results.json"
    with open(output_filename, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved integrated results to {output_filename}")

if __name__ == "__main__":
    main()
