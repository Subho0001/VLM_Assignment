import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

class VLMEngine:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM-500M-Instruct"):
        print("Loading VLM Engine...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()

    def analyze_document(self, image, task_type):
        """Routes the image to specific prompts based on the required task."""
        
        # --- PROMPT ENGINEERING THE TASKS ---
        if task_type == "extraction":
            prompt = (
                "You are a precise data extraction system. Read this receipt and extract the "
                "Total, Tax, and Menu Items. Output the result strictly as a JSON dictionary. "
                "Do not include markdown, code blocks, or conversational text."
            )
        elif task_type == "signature":
            prompt = (
                "Look carefully at this document. Is there a handwritten signature present? "
                "Answer with exactly one word: 'Yes' or 'No'."
            )
        elif task_type == "form_fields":
            prompt = (
                "Identify all form fields, checkboxes, or input lines in this document. "
                "Output a JSON list where each item has the 'field_name' and a 'status' "
                "of either 'filled' or 'empty'."
            )
        else:
            prompt = "Describe this document."

        # --- INFERENCE ---
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False, # STRICT DETERMINISM: Prevents hallucinations
            )

        input_len = inputs["input_ids"].shape[1]
        output_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]

        return output_text.strip()
