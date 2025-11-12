import json
import re

def extract_method(source_code, method_name):
    """
    Extracts a method's source code from a larger script based on its name and indentation.
    Handles optional @staticmethod decorators.
    """
    lines = source_code.split('\n')
    method_lines = []
    in_method = False
    method_indentation = -1
    start_line_index = -1

    # Find the start of the method definition
    for i, line in enumerate(lines):
        stripped_line = line.lstrip()
        if stripped_line.startswith(f"def {method_name}("):
            in_method = True
            method_indentation = len(line) - len(stripped_line)
            start_line_index = i
            break

    if not in_method:
        return None  # Method not found

    # Once the method is found, collect its lines
    # Check for a decorator on the line above
    if start_line_index > 0 and lines[start_line_index - 1].lstrip().startswith('@staticmethod'):
        method_lines.append(lines[start_line_index - 1])

    method_lines.append(lines[start_line_index])

    # Collect subsequent lines that are part of the method body
    for i in range(start_line_index + 1, len(lines)):
        line = lines[i]
        stripped_line = line.lstrip()

        # A line is part of the method if it's empty or more indented
        if not stripped_line or (len(line) - len(stripped_line)) > method_indentation:
            method_lines.append(line)
        else:
            # Reached a line that is not part of the method (less or equal indentation)
            break

    return '\n'.join(method_lines)

def refactor_notebook(notebook_path):
    """
    Refactors a Jupyter notebook by moving specified methods into a new Visualizer class.
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except FileNotFoundError:
        return f"Error: Notebook file not found at {notebook_path}"

    # Target the specific cell containing the classes
    if len(notebook['cells']) < 3:
        return "Error: Notebook does not have enough cells."
    cell = notebook['cells'][2]
    source_code = '\n'.join(cell['source'])

    visualizer_class_code = 'class Visualizer(object):'

    methods_to_move = [
        '_trend_check',
        'display_causalimpact_result',
        'plot_causalimpact',
        '_visualize_candidate',
        '_display_simulation_result',
        '_plot_simulation_result'
    ]

    methods_code = []
    for method_name in methods_to_move:
        method_code = extract_method(source_code, method_name)
        if method_code is None:
            print(f"Error: Method '{method_name}' not found. Aborting.")
            return

        methods_code.append(method_code)
        source_code = source_code.replace(method_code, '')

    for method_code in methods_code:
        indented_method_code = '\n  ' + method_code.replace('\n', '\n  ')
        visualizer_class_code += indented_method_code
    visualizer_class_code += '\n'

    # If an empty Visualizer class exists, replace it. Otherwise, insert the new class.
    if "class Visualizer(object):" in source_code:
        # Use a regex to replace the (potentially empty) class definition
        source_code = re.sub(r'class Visualizer\(object\):(\s*\\n)*', visualizer_class_code, source_code, count=1)
    else:
        source_code = source_code.replace(
            'class CausalImpact(PreProcess):',
            visualizer_class_code + '\nclass CausalImpact(PreProcess):',
            1
        )

    # Clean up excessive blank lines
    source_code = re.sub(r'\n\s*\n\s*\n', '\n\n', source_code)

    # Update the cell source
    cell['source'] = source_code.splitlines(True)

    try:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
    except IOError as e:
        return f"Error writing to notebook file: {e}"

    return "Refactoring for Visualizer class complete."

if __name__ == '__main__':
    result = refactor_notebook('solutions/causal-impact/CausalImpact_with_Experimental_Design.ipynb')
    print(result)
