import json
import re

def add_delegation_methods(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cell = notebook['cells'][2]
    source_code = '\n'.join(cell['source'])

    # Add delegation methods to CausalImpactAnalysis
    delegation_methods = """
  def run_causal_impact(self):
    return self.model.run_causal_impact(self)

  def run_experimental_design(self):
    return self.experimental_designer.run_experimental_design(self)

  def generate_simulation(self):
    return self.simulator.generate_simulation(self)

  def display_causalimpact_result(self):
    return self.visualizer.display_causalimpact_result(self)
"""
    source_code = source_code.replace(
        "class CausalImpactAnalysis(PreProcess):",
        "class CausalImpactAnalysis(PreProcess):" + delegation_methods
    )

    # Update method signatures in helper classes to accept the orchestrator instance
    # This is a simplified approach; a more robust solution would use AST parsing
    source_code = re.sub(r'def run_causal_impact\(self\):', r'def run_causal_impact(self, orchestrator):', source_code)
    source_code = re.sub(r'def run_experimental_design\(self\):', r'def run_experimental_design(self, orchestrator):', source_code)
    source_code = re.sub(r'def generate_simulation\(self\):', r'def generate_simulation(self, orchestrator):', source_code)
    source_code = re.sub(r'def display_causalimpact_result\(self\):', r'def display_causalimpact_result(self, orchestrator):', source_code)

    # Update calls within the helper methods to use the orchestrator instance
    source_code = source_code.replace('self.formatted_data', 'orchestrator.formatted_data')
    source_code = source_code.replace('self.date_col_name', 'orchestrator.date_col_name')
    source_code = source_code.replace('self.pre_period_start.value', 'orchestrator.pre_period_start.value')
    source_code = source_code.replace('self.pre_period_end.value', 'orchestrator.pre_period_end.value')
    source_code = source_code.replace('self.post_period_start.value', 'orchestrator.post_period_start.value')
    source_code = source_code.replace('self.post_period_end.value', 'orchestrator.post_period_end.value')
    source_code = source_code.replace('self.num_of_seasons.value', 'orchestrator.num_of_seasons.value')
    source_code = source_code.replace('self.credible_interval.value', 'orchestrator.credible_interval.value')
    source_code = source_code.replace('self.date_selection.selected_index', 'orchestrator.date_selection.selected_index')
    source_code = source_code.replace('self.start_date.value', 'orchestrator.start_date.value')
    source_code = source_code.replace('self.end_date.value', 'orchestrator.end_date.value')
    source_code = source_code.replace('self.design_type.selected_index', 'orchestrator.design_type.selected_index')
    source_code = source_code.replace('self.num_of_split.value', 'orchestrator.num_of_split.value')
    source_code = source_code.replace('self.target_columns.value', 'orchestrator.target_columns.value')
    source_code = source_code.replace('self.num_of_pick_range.value', 'orchestrator.num_of_pick_range.value')
    source_code = source_code.replace('self.num_of_covariate.value', 'orchestrator.num_of_covariate.value')
    source_code = source_code.replace('self.target_share.value', 'orchestrator.target_share.value')
    source_code = source_code.replace('self.control_columns.value', 'orchestrator.control_columns.value')
    source_code = source_code.replace('self.tick_count', 'orchestrator.tick_count')
    source_code = source_code.replace('self.your_choice.value', 'orchestrator.your_choice.value')
    source_code = source_code.replace('self.target_col_to_simulate.value', 'orchestrator.target_col_to_simulate.value')
    source_code = source_code.replace('self.covariate_col_to_simulate.value', 'orchestrator.covariate_col_to_simulate.value')
    source_code = source_code.replace('self.distance_data', 'orchestrator.distance_data')
    source_code = source_code.replace('self.start_date_value', 'orchestrator.start_date_value')
    source_code = source_code.replace('self.end_date_value', 'orchestrator.end_date_value')
    source_code = source_code.replace('self.estimate_icpa.value', 'orchestrator.estimate_icpa.value')
    source_code = source_code.replace('self.purpose_selection.selected_index', 'orchestrator.purpose_selection.selected_index')
    source_code = source_code.replace('self.ci_objs', 'orchestrator.ci_objs')

    cell['source'] = source_code.splitlines(True)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    return "Delegation methods added and helper method signatures updated."

if __name__ == '__main__':
    result = add_delegation_methods('solutions/causal-impact/CausalImpact_with_Experimental_Design.ipynb')
    print(result)
