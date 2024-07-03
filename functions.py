from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from datasets import load_dataset
import soundfile as sf
import torch
import subprocess
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

device = "cpu"


def image2text(image_path):

    try:
        image = Image.open(image_path).convert("RGB")

        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text

    except Exception as e:
        print(f"Error during image processing: {e}")
        return None


def textify(text):

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    if text:

        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        generation_prompt = f"Correct any mistakes in the following text: {text}"
        generation_outputs = model.generate(**inputs, prompt=generation_prompt)

        final_text = tokenizer.decode(generation_outputs[0], skip_special_tokens=True)

    else:

        final_text = "There is no text available"

    return final_text


def text2audio(text):

    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

    return speech