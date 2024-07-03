import torch
from tqdm import tqdm
from model_loader import load_model
from utils import load_jsonl, add_jsonl
import numpy as np
import zlib


def calculate_likelihood(sentence: str, model, tokenizer, device):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    del input_ids
    del logits
    torch.cuda.empty_cache()

    return torch.exp(loss).item(), all_prob, loss.item()


def main():
    model_name = "gpt-j-6B"  # gpt-j-6B, opt-6.7b, pythia-6.9b, Llama-2-7b
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model, tokenizer = load_model(model_name, device=device)

    for text_length in [32, 64, 128, 256]:
        lines = load_jsonl(f'wikimia/{text_length}.jsonl')
        for line in tqdm(lines):
            text = line["input"]
            ppl, all_prob, loss = calculate_likelihood(text, model, tokenizer, device=device)
            new_line = {
                "text": text,
                "label": line["label"],
                "perplexity": ppl,
                "all_prob": all_prob,
                "loss": loss,
            }
            # min-k prob
            ratio = 0.2
            k_length = int(len(all_prob)*ratio)
            topk_prob = np.sort(all_prob)[:k_length]
            new_line[f"min20%prob"] = np.mean(topk_prob).item()

            # lowercase
            ppl_lower, _, _ = calculate_likelihood(text.lower(), model, tokenizer, device=device)
            new_line["ppl/ppl_lower"] = ppl/ppl_lower

            # zlib
            zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
            new_line["ppl/zlib"] = ppl/zlib_entropy

            add_jsonl(new_line, f'likelihood/{model_name}/{text_length}.jsonl')


if __name__ == '__main__':
    main()