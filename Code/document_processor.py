import os
from pdf2image import convert_from_path
from PIL import Image

def process_pdf_to_images(pdf_path, dpi=200):
    """
    Converts a multi-page PDF into a list of PIL Images.
    Resizes them to prevent VRAM overflow during VLM inference.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    print(f"Converting {pdf_path} to images...")
    pages = convert_from_path(pdf_path, dpi=dpi)

    processed_images = []
    for i, page in enumerate(pages):
        # Ensure RGB mode
        if page.mode != 'RGB':
            page = page.convert('RGB')

        # Resize to max 800px dimension to save VRAM while preserving OCR quality
        w, h = page.size
        max_dim = 800
        if max(w, h) > max_dim:
            ratio = max_dim / float(max(w, h))
            page = page.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

        processed_images.append(page)
        print(f"  -> Processed Page {i+1}")

    return processed_images
