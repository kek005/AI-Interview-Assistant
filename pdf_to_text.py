import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_file, output_txt_path):
    """Extract text from a PDF and save it as a .txt file."""
    doc = fitz.open(pdf_file)
    with open(output_txt_path, "w", encoding="utf-8") as txt_file:
        for page in doc:
            text = page.get_text()
            txt_file.write(text + "\n\n")  # Write each page's text

    print(f"Text extracted and saved to {output_txt_path}")

# Example Usage
pdf_file = r"/home/keke/2510_Python-Machine-Learning-Projects.pdf"  # Change this to your actual PDF file
txt_output = "/home/keke/python_ml_course.txt"
extract_text_from_pdf(pdf_file, txt_output)