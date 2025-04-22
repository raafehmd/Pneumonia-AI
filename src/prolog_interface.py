# prolog_interface.py
import subprocess
import os

def run_prolog_diagnosis(has_pneumonia: str):
    rules_path = os.path.join(os.path.dirname(__file__), 'rules.pl')
    query = f"has_pneumonia({has_pneumonia}), halt."
    result = subprocess.run(
        ['swipl', '-s', rules_path, '-g', query],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()
