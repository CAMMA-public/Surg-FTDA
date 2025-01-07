from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
# Load the JSON file
input_json = "/home/ubuntu/meddataset/choelc_generate/phase_test/v2/clip/inference_noise_0015/translations_final.json"

# Load the data
with open(input_json, "r") as f:
    data = json.load(f)

# Initialize lists to store generated and reference phases
generated_phases = []
reference_phases = []

# Extract the phases from the JSON structure
for entry in data["translations"]:

    generated_phase = entry["generated"].replace("phases: ", "")  # Remove the 'phases: ' prefix
    reference_phase = entry["reference"]["phases"].replace("phases: ", "")  # Remove the 'phases: ' prefix
    
    generated_phases.append(generated_phase)
    reference_phases.append(reference_phase)

# Calculate metrics
print(generated_phases[:10])
print(reference_phases[:10])
acc = accuracy_score(reference_phases, generated_phases)
precision = precision_score(reference_phases, generated_phases, average='macro', zero_division=0)
recall = recall_score(reference_phases, generated_phases, average='macro', zero_division=0)
f1 = f1_score(reference_phases, generated_phases, average='macro', zero_division=0)

# Print the results
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Optionally, you can uncomment the confusion matrix section if needed
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(reference_phases, generated_phases)
# print("Confusion Matrix:")
# print(cm)

from sklearn.metrics import classification_report
print(classification_report(reference_phases, generated_phases, zero_division=0))
