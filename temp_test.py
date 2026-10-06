from pathlib import Path

from llama_index.readers.file import PyMuPDFReader

pdf_path = Path("docs/RAG/NIST.SP.800-53r5.pdf")

reader = PyMuPDFReader()

documents = reader.load(file_path=pdf_path)

print(f"Documents/pages returned: {len(documents)}")

for index, document in enumerate(
    documents[:3],
    start=1,
):
    text = document.get_content().strip()

    print("\n" + "=" * 70)
    print(f"DOCUMENT {index}: {len(text)} chars")
    print("=" * 70)
    print(text[:1000])
