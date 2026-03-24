import gradio as gr
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from gpt import GPTWithStyle, _apply_sampling

device = 'cpu'

with open('corpus_clean.txt', 'r') as f:
    text = f.read()

characters = sorted(list(set(text)))
vocab_size = len(characters)
char_to_idx = {ch: i for i, ch in enumerate(characters)}
idx_to_char = {i: ch for i, ch in enumerate(characters)}
encode = lambda xs: [char_to_idx[x] for x in xs if x in char_to_idx]
decode = lambda xs: ''.join([idx_to_char[x] for x in xs])

model_path = hf_hub_download(repo_id='HAR5HA-YELLELA/luffy-gpt', filename='luffy_gpt_finetuned.pth')
model = GPTWithStyle(vocab_size, n_embd=384, context_size=256, n_head=6, n_layer=6, n_styles=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

style_t = torch.tensor([1], dtype=torch.long, device=device)
eos_tokens = encode('<EOS>')


def chat(user_message):
    prompt = f'USER: {user_message}\nLUFFY:'
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, style_t, number_of_tokens=150,
                         temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.2,
                         eos_tokens=eos_tokens)
    result = decode(out[0].tolist())
    result = result.split('<EOS>')[0].split('\nUSER:')[0].strip()
    return result


demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label='Ask Luffy', placeholder='Who are you?', lines=2),
    outputs=gr.Textbox(label='Luffy says', lines=5),
    title='Luffy GPT',
    description='10.8M param GPT trained from scratch on One Piece dialogue, fine-tuned with 5k SFT pairs. Not a perfect chatbot -- works best with direct questions.',
    examples=[
        ['Who are you?'],
        ['What is your dream?'],
        ['Are you hungry?'],
        ['Are you scared?'],
        ['Who is Zoro?'],
        ["What's your favourite food?"],
        ['Who is Shanks?'],
        ['What happened to Ace?'],
        ['Are you a hero?'],
        ['Do you like parties?'],
    ]
)

if __name__ == '__main__':
    demo.launch()
