from docx import Document
doc = Document(r'C:\Users\Parhavi G.V\Downloads\survey paper capstone.docx')
for i,p in enumerate(doc.paragraphs):
    if p.text.strip():
        print(p.text[:400])
    if i > 200:
        break
