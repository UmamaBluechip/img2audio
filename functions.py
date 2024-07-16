from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2Model, AutoProcessor
from PIL import Image
from datasets import load_dataset
import soundfile as sf
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image2text(image_path):

    try:
        prompt = "<OCR>"
        model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True).to(device).eval()
        processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)
        
        image = Image.open(image_path)
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        text = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
        
        return text

    except Exception as e:
        print(f"Error during image processing: {e}")
        return None


def textify(text):

    #model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    #tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    
    if text:

        prompt = f"Correct any mistakes in the following text: {text}"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        #generation_prompt = f"Correct any mistakes in the following text: {text}"
        #generation_outputs = model.generate(**inputs, prompt=generation_prompt)

        #final_text = tokenizer.decode(generation_outputs[0], skip_special_tokens=True)

        final_text = model(**inputs)

    else:

        final_text = "There is no text available"

    return final_text


def text2audio(text):

    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

    return speech