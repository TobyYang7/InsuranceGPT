from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/zhangmin/toby/IBA_Project_24spr/saves/igpt_v1"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)

system_message = "You are a helpful assistant. You should guide the user to buy the insurance product."

print("Chat with the model. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print("Model:", tokenizer.decode(response, skip_special_tokens=True))
