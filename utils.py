
import sentencepiece as spm
import torch
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent

# Load config
with open(BASE / "config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

PAD = cfg["pad_id"]
UNK = cfg["unk_id"]
SOS = cfg["sos_id"]
EOS = cfg["eos_id"]

# Load tokenizers
ur_tok = spm.SentencePieceProcessor(model_file=str(BASE / "urdu_bpe.model"))
en_tok = spm.SentencePieceProcessor(model_file=str(BASE / "eng_bpe.model"))

def encode_text(text: str, tokenizer, max_len=100):
    ids = tokenizer.encode(text, out_type=int)[:max_len-2]
    ids = [SOS] + ids + [EOS]
    return torch.tensor([ids], dtype=torch.long)

def decode_ids(ids):
    # ids: list or 1D tensor of ints (no batch)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    # remove sos/eos if present
    clean = []
    for i in ids:
        if i == SOS:
            continue
        if i == EOS:
            break
        clean.append(i)
    return en_tok.decode(clean)

def greedy_translate(model, src_text, device="cpu", max_len=100):
    model.eval()
    src_tensor = encode_text(src_text, ur_tok, max_len=max_len).to(device)
    with torch.no_grad():
        # Run encoder
        _, hidden, cell = model.encoder(src_tensor)
        # start token
        cur_token = torch.tensor([[SOS]], dtype=torch.long).to(device)
        output_ids = []
        for _ in range(max_len):
            preds, hidden, cell = model.decoder(cur_token, hidden, cell)  # preds: [1,1,V]
            preds = preds.squeeze(0).squeeze(0)  # [V]
            next_id = int(torch.argmax(preds).item())
            if next_id == EOS:
                break
            output_ids.append(next_id)
            cur_token = torch.tensor([[next_id]], dtype=torch.long).to(device)
    return decode_ids(output_ids)
