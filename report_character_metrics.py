import pandas as pd
import editdistance
import os

RESULTS_CSV = "evaluation_results.csv"

def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"❌ Not found: {RESULTS_CSV}")
        print("Run evaluate.py first to generate evaluation_results.csv")
        return

    df = pd.read_csv(RESULTS_CSV)

    if "predicted" not in df.columns or "ground_truth" not in df.columns:
        print("❌ CSV must contain columns: predicted, ground_truth")
        return

    preds = df["predicted"].fillna("").astype(str).tolist()
    gts = df["ground_truth"].fillna("").astype(str).tolist()

    n = len(preds)
    if n == 0:
        print("❌ Empty evaluation_results.csv")
        return

    exact = sum(p == g for p, g in zip(preds, gts))
    wer = (n - exact) / n * 100.0

    total_chars = sum(len(g) for g in gts)
    edits = sum(editdistance.eval(p, g) for p, g in zip(preds, gts))
    cer = (edits / total_chars * 100.0) if total_chars > 0 else 0.0

    # Character Accuracy = 1 - CER
    char_acc = 100.0 - cer

    print("\n" + "=" * 55)
    print(f"Samples: {n}")
    print(f"Exact Match Accuracy: {exact / n * 100:.2f}%")
    print(f"Word Error Rate (WER): {wer:.2f}%")
    print(f"Character Error Rate (CER): {cer:.2f}%")
    print(f"Character Accuracy: {char_acc:.2f}%")
    print("=" * 55)

if __name__ == "__main__":
    main()
