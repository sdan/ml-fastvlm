""" Given a list of PDF files, first split the pdfs into sections, then embed the sections agaist a list of categories.
"""
from ast import List
import os

class PDFCollater:
    def __init__(self, pdf_files: List[str]):
        self.pdf_files = pdf_files
    
    def split_pdfs(self):
        for pdf_file in self.pdf_files:
            with open(pdf_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    print(text)
    
    def embed_sections(self):
        pass

def main():
    pass


if __name__ == "__main__":
    main()
