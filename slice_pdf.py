import pypdfium2 as pdfium
import sys

# Load the original PDF
pdf = pdfium.PdfDocument("test_document.pdf")

# Create a new blank PDF
new_pdf = pdfium.PdfDocument.new()

# Extract the first 4 pages (pages 0, 1, 2, 3) which contain the abstract, intro, background, and the main architecture diagram & equations
new_pdf.import_pages(pdf, [0, 1, 2, 3])

# Save it as test_document_short.pdf
new_pdf.save("test_document_short.pdf")
print("Successfully created test_document_short.pdf")



