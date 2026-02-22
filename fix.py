import nbformat

filename = r"C:\Users\PC\Desktop\summative\Domain-Specific-Assistant-via-LLMs-Fine-Tunin\Notebook\Customer_Support_Chatbot.ipynb"

with open(filename, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

with open(filename, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Fixed! Outputs preserved.")