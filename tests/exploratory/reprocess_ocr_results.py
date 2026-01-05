"""Re-process OCR results with improved classifier."""
import pandas as pd
from src.workzone.ocr.text_classifier import TextClassifier

# Load intensive test results
df = pd.read_csv('outputs/ocr_intensive_test_results.csv')

# Filter only detected cases
detected = df[df['detection_success'] == True].copy()
print(f"Total detections: {len(detected)}")
print(f"Before: UNCLEAR = {(detected['text_category'] == 'UNCLEAR').sum()} / {len(detected)} = {100*(detected['text_category'] == 'UNCLEAR').sum()/len(detected):.1f}%\n")

# Re-classify with improved classifier
classifier = TextClassifier()

new_categories = []
new_confidences = []

for text in detected['ocr_text']:
    category, confidence = classifier.classify(text)
    new_categories.append(category)
    new_confidences.append(confidence)

detected['text_category_new'] = new_categories
detected['category_confidence_new'] = new_confidences

# Compare results
print("=" * 80)
print("IMPROVEMENT ANALYSIS")
print("=" * 80)

# Category distribution
print("\nCategory Distribution:")
print("BEFORE (old classifier):")
print(detected['text_category'].value_counts())
print("\nAFTER (improved classifier):")
print(detected['text_category_new'].value_counts())

# Calculate improvements
unclear_before = (detected['text_category'] == 'UNCLEAR').sum()
unclear_after = (detected['text_category_new'] == 'UNCLEAR').sum()
noise_after = (detected['text_category_new'] == 'NOISE').sum()

useful_before = len(detected) - unclear_before
useful_after = len(detected) - unclear_after - noise_after

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print(f"UNCLEAR cases:    {unclear_before} → {unclear_after}  ({unclear_before - unclear_after:+d})")
print(f"NOISE detected:   0 → {noise_after}  (+{noise_after})")
print(f"Useful detections: {useful_before} → {useful_after}  ({useful_after - useful_before:+d})")
print()
print(f"Useful rate: {100*useful_before/len(detected):.1f}% → {100*useful_after/len(detected):.1f}%  ({100*(useful_after-useful_before)/len(detected):+.1f}%)")

# Show examples of improvements
print("\n" + "=" * 80)
print("EXAMPLES OF IMPROVEMENTS (UNCLEAR → Classified):")
print("=" * 80)

improved = detected[(detected['text_category'] == 'UNCLEAR') & (detected['text_category_new'] != 'UNCLEAR')][:20]
for idx, row in improved.iterrows():
    print(f"✅ '{row['ocr_text'][:50]}'")
    print(f"   OLD: UNCLEAR → NEW: {row['text_category_new']} ({row['category_confidence_new']:.2f})")

# Show examples of noise detection
print("\n" + "=" * 80)
print("EXAMPLES OF NOISE FILTERING (Previously Classified → NOISE):")
print("=" * 80)

noise_cases = detected[(detected['text_category'] != 'UNCLEAR') & (detected['text_category_new'] == 'NOISE')][:15]
for idx, row in noise_cases.iterrows():
    print(f"🗑️  '{row['ocr_text'][:50]}'")
    print(f"   OLD: {row['text_category']} → NEW: NOISE (correctly filtered)")

# Save results
output_file = 'outputs/ocr_reprocessed_improved.csv'
detected.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")
