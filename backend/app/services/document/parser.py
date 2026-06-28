import fitz  # PyMuPDF for PDF
from typing import List
from docx import Document as DocxDocument
import csv

class PDFParser:
    """Parse PDF files"""
    
    @staticmethod
    def parse(file_path: str) -> str:
            text = ""
            try:
                doc = fitz.open(file_path)
                for page in doc:
                    page_text = page.get_text()
                    if page_text.strip():
                        text += page_text + "\n"
                doc.close()
                # DEBUG - remove after testing
                print(f"[PARSER DEBUG] Total chars: {len(text)}")
                print(f"[PARSER DEBUG] Newline count: {text.count(chr(10))}")
                print(f"[PARSER DEBUG] First 300 chars: {repr(text[:300])}")
                return text
            except Exception as e:
                raise ValueError(f"Error parsing PDF: {str(e)}")
        


class DOCXParser:
    """Parse DOCX files"""
    
    @staticmethod
    def parse(file_path: str) -> str:
        """Extract text from DOCX"""
        text = ""
        try:
            doc = DocxDocument(file_path)
            
            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            
            # Extract tables
            for table in doc.tables:
                text += "\n[TABLE]\n"
                for row in table.rows:
                    row_text = " | ".join(cell.text for cell in row.cells)
                    text += row_text + "\n"
                text += "[/TABLE]\n"
            
            return text
        except Exception as e:
            raise ValueError(f"Error parsing DOCX: {str(e)}")

import csv

class TXTParser:
    """Parse plain text files"""
    
    @staticmethod
    def parse(file_path: str) -> str:
        """Read text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error parsing TXT: {str(e)}")

class CSVParser:
    """Parse CSV files"""
    
    @staticmethod
    def parse(file_path: str) -> str:
        """Convert CSV to text format"""
        text = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        text += "HEADERS: " + " | ".join(row) + "\n"
                    else:
                        text += " | ".join(row) + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error parsing CSV: {str(e)}")

# Main parser dispatcher
class DocumentParser:
    """Main document parser"""
    
    PARSERS = {
        "pdf": PDFParser,
        "docx": DOCXParser,
        "txt": TXTParser,
        "csv": CSVParser
    }
    
    @staticmethod
    def parse(file_path: str, file_type: str) -> str:
        """Parse document based on type"""
        file_type = file_type.lower()
        if file_type not in DocumentParser.PARSERS:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        parser_class = DocumentParser.PARSERS[file_type]
        return parser_class.parse(file_path)