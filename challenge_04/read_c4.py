from pypdf import PdfReader

reader = PdfReader("/Users/raunavmendiratta/Downloads/challenge (2).pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

try:
    idx = text.find("Challenge 4")
    if idx != -1:
        print(text[idx:idx+500])
    else:
        print("Challenge 4 not found in text.")
        print(text) # Dump all to find it manually
except Exception as e:
    print(e)