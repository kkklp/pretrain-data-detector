from sklearn.metrics import roc_curve, auc
from utils import load_jsonl


def calculate_auc(method, model_name, text_length):
    lines = load_jsonl(f'likelihood/{model_name}/{text_length}.jsonl')
    data_seen = [line[method] for line in lines if line["label"] == 1]
    data_unseen = [line[method] for line in lines if line["label"] == 0]

    if method in {"loss", "ppl/ppl_lower", "ppl/zlib"}:
        data_big = data_unseen
        data_small = data_seen
    elif method in {"min20%prob"}:
        data_big = data_seen
        data_small = data_unseen
    
    y_true = [1] * len(data_big) + [0] * len(data_small)
    y_score = data_big + data_small
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    return auc_score


def main():
    model_name = "gpt-j-6B"  # gpt-j-6B, opt-6.7b, pythia-6.9b, Llama-2-7b
    text_length = 32  # 32, 64, 128, 256

    for method in ["loss", "ppl/ppl_lower", "ppl/zlib", "min20%prob"]:
        auc = calculate_auc(method, model_name, text_length)
        print(f"{method}: {auc:.2f}")


if __name__ == "__main__":
    main()
