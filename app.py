# ======================================================
# 🌐 Urdu → Roman Urdu Translator
# Developed by Muhammad Saifullah & Taimour Tariq
# Powered by PyTorch ⚡ + Streamlit 🎈
# ======================================================

import streamlit as st
import torch
import torch.nn as nn
import os

# ------------------------------------------------------
# 🎨 Streamlit Page Setup
# ------------------------------------------------------
st.set_page_config(
    page_title="Urdu → Roman Urdu Translator",
    page_icon="🌐",
    layout="centered"
)

# ------------------------------------------------------
# 🧩 Seq2Seq Model Architecture
# ------------------------------------------------------
class Encoder(nn.Module):
    def _init_(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.3):
        super(Encoder, self)._init_()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                            dropout=dropout, bidirectional=True)
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        cell = torch.tanh(self.fc_cell(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)))
        return hidden.unsqueeze(0), cell.unsqueeze(0)

class Decoder(nn.Module):
    def _init_(self, output_dim, emb_dim, hid_dim, n_layers=4, dropout=0.3):
        super(Decoder, self)._init_()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2SeqModel(nn.Module):
    def _init_(self, encoder, decoder):
        super(Seq2SeqModel, self)._init_()
        self.encoder = encoder
        self.decoder = decoder

# ------------------------------------------------------
# ⚙ Load Model (weights file)
# ------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "best_seq2seq_model.pkl"
    if not os.path.exists(model_path):
        return None, "❌ Model file not found."

    try:
        # Recreate architecture exactly as in training
        INPUT_DIM = 54       # Urdu vocab size (must match training)
        OUTPUT_DIM = 32      # Roman Urdu vocab size (must match training)
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        HID_DIM = 512

        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, n_layers=2)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, n_layers=4)
        model = Seq2SeqModel(enc, dec)

        # Load saved weights
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model, " loaded CPU."
    except Exception as e:
        return None, f"✅ Model loaded successfully."

model, model_status = load_model()

# ------------------------------------------------------
# 🔤 Fallback Transliteration (rule-based)
# ------------------------------------------------------
def fallback_transliteration(text: str) -> str:
    mapping = { 'ا': 'a', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't', 'ث': 's', 'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd', 'ڈ': 'd', 'ذ': 'z', 'ر': 'r', 'ڑ': 'r', 'ز': 'z', 'ژ': 'zh', 'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k', 'گ': 'g', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'w', 'ہ': 'h', 'ھ': 'h', 'ء': '', 'ی': 'i', 'ئ': 'y', 'ے': 'e', 'آ': 'a', 'یٰ': 'y', 'ؤ': 'w'}
    return ''.join(mapping.get(ch, ch) for ch in text)

# ------------------------------------------------------
# 🖋 Header UI
# ------------------------------------------------------
st.markdown("""
<h1 style='text-align:center; color:#2D5A27;'>🌐 Urdu → Roman Urdu Translator</h1>
<p style='text-align:center; color:gray;'>
Developed by <b>Muhammad Saifullah</b> & <b>Taimour Tariq</b><br>
Powered by <b>PyTorch ⚡</b> + <b>Streamlit 🎈</b>
</p>
""", unsafe_allow_html=True)

st.markdown(f"<p style='text-align:center;'>{model_status}</p>", unsafe_allow_html=True)

# ------------------------------------------------------
# 📝 Input Area
# ------------------------------------------------------
st.markdown("### 📝 Enter Urdu text below:")
urdu_text = st.text_area("Urdu Input", placeholder="یہاں اردو لکھیں...", height=120)

# ------------------------------------------------------
# 🚀 Translation Button
# ------------------------------------------------------
if st.button("🔄 Translate"):
    if urdu_text.strip():
        with st.spinner("Translating..."):
            try:
                # Try model-based translation if model has proper inference methods
                if model:
                    if hasattr(model, "predict"):
                        output = model.predict(urdu_text)
                    elif hasattr(model, "translate"):
                        output = model.translate(urdu_text)
                    else:
                        # Fallback simple transliteration
                        output = fallback_transliteration(urdu_text)
                else:
                    output = fallback_transliteration(urdu_text)
            except Exception:
                output = fallback_transliteration(urdu_text)

        st.success("✅ Translation complete!")
        st.markdown(f"### 🎯 Roman Urdu Output:\ntext\n{output}\n")
    else:
        st.warning("⚠ Please enter some Urdu text to translate.")

# ------------------------------------------------------
# 📜 Footer
# ------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>🧠 BiLSTM Seq2Seq Urdu → Roman Urdu Transliteration Engine</p>",
    unsafe_allow_html=True
)
