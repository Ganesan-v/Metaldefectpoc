import os

def save_flagged(file):
    os.makedirs("output/flagged_faulty", exist_ok=True)
    with open(f"output/flagged_faulty/{file.name}", "wb") as f:
        f.write(file.read())

def generate_report(results):
    os.makedirs("reports", exist_ok=True)
    with open("reports/analysis_report.txt", "w") as f:
        f.write("Metal Defect Detection Summary Report\n")
        f.write(f"Total Images Analyzed: {results['total']}\n")
        f.write(f"Faulty Metals: {results['faulty']}\n")
        f.write(f"Quality OK Metals: {results['quality_ok']}\n")
