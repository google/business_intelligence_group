import json

notebook_path = 'solutions/causal-impact/CausalImpact_with_Experimental_Design.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = cell['source']

        # Find the start of the generate_ui method
        for i, line in enumerate(source):
            if 'def generate_ui(self):' in line:
                # The unwanted block is at a fixed position relative to this line
                # It starts at i + 5 and ends at i + 28.
                # Let's add some checks to be safe.
                if (i + 28 < len(source) and
                    source[i+5].strip() == "" and
                    'display(' in source[i+6] and 'simulation_df.query' in source[i+6]):

                    # The block to remove is from line i+5 to i+28 inclusive.
                    # This corresponds to a slice of length 24.
                    del source[i+5 : i+29]
                break

        cell['source'] = source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Cleaned up generate_ui method in PreProcess class.")
