
# example_load.py (for serve/)
import torch
from model import Encoder, Decoder, Seq2Seq
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
with open(BASE / "config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = Encoder(cfg["input_dim"], cfg["embed_dim"], cfg["hidden_dim"], n_layers=cfg["enc_layers"], dropout=0.3)
dec = Decoder(cfg["output_dim"], cfg["embed_dim"], cfg["hidden_dim"], n_layers=cfg["dec_layers"], dropout=0.3)
model = Seq2Seq(enc, dec, device).to(device)
state = torch.load(BASE / "best_seq2seq_model.pth", map_location=device)
model.load_state_dict(state)
model.eval()
print("Model loaded for inference")
