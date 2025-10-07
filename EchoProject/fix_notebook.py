import nbformat

# Replace 'your_notebook.ipynb' with your actual notebook filename
notebook_name = 'your_notebook.ipynb'

# Read the notebook
with open(notebook_name, 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Remove widget metadata
if 'widgets' in nb.metadata:
    del nb.metadata['widgets']
    print("Removed widget metadata")
else:
    print("No widget metadata found")

# Save the fixed notebook
with open(notebook_name, 'w') as f:
    nbformat.write(nb, f)

print(f"Fixed {notebook_name}!")
