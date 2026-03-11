import gradio as gr
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from gpt import GPT

device = 'cpu'

with open('corpus_clean.txt', 'r') as f:
    text = f.read()

characters = sorted(list(set(text)))
vocab_size = len(characters)
char_to_idx = {ch: i for i, ch in enumerate(characters)}
idx_to_char = {i: ch for i, ch in enumerate(characters)}
encode = lambda xs: [char_to_idx[x] for x in xs if x in char_to_idx]
decode = lambda xs: ''.join([idx_to_char[x] for x in xs])

model_path = hf_hub_download(repo_id='HAR5HA-YELLELA/luffy-gpt', filename='luffy_gpt.pth')
model = GPT(vocab_size, n_embd=384, context_size=256, n_head=6, n_layer=6).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def generate(prompt, num_tokens):
    if prompt:
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(int(num_tokens)):
            idx_cond = idx[:, -256:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return decode(idx[0].tolist())


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label='Prompt', placeholder='Luffy:', lines=2),
        gr.Slider(50, 500, value=200, step=50, label='Tokens to generate'),
    ],
    outputs=gr.Textbox(label='Generated text', lines=10),
    title='Luffy GPT',
    description='GPT trained on One Piece dialogue. Type a prompt and see what happens.',
    examples=[
        ['Luffy:', 200],
        ['Zoro:', 200],
        ['Nami:', 200],
    ]
)

if __name__ == '__main__':
    demo.launch()
