import re

def clean_text(input_txt_path, output_cleaned_txt):
    """Cleans extracted text from a PDF file."""
    
    with open(input_txt_path, "r", encoding="utf-8") as file:
        text = file.readlines()
    
    cleaned_lines = []
    
    for line in text:
        # Remove lines with page numbers, headers, excessive spaces
        if re.search(r'^[0-9]+$', line.strip()):  # Remove lines that are just numbers (page numbers)
            continue
        if "Copyright" in line or "All rights reserved" in line:  # Remove copyright info
            continue
        
        # Remove multiple spaces, normalize
        line = re.sub(r'\s+', ' ', line)  # Replace multiple spaces with a single space
        line = line.strip()  # Remove trailing spaces
        if line:  # Skip empty lines
            cleaned_lines.append(line)

    # Save cleaned text
    with open(output_cleaned_txt, "w", encoding="utf-8") as file:
        file.write("\n".join(cleaned_lines))
    
    print(f"Cleaned text saved to {output_cleaned_txt}")

# Example Usage
raw_text_file = "/home/keke/python_ml_course.txt"
cleaned_text_file = "/home/keke/python_ml_course_cleaned.txt"

clean_text(raw_text_file, cleaned_text_file)