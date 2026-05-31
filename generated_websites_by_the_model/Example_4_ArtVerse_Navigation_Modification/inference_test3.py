#pip install deep-translator

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList
#from deep_translator import GoogleTranslator

# ========= 1️⃣ تحميل أفضل checkpoint =========
model_path = "./results/checkpoint-3680"   # ضع هنا أفضل نقطة تحقق عندك

tokenizer = AutoTokenizer.from_pretrained("./results")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

class StopOnHTMLClose(StoppingCriteria):
          def __init__(self, tokenizer, stop_str="</html>"):
              self.tokenizer = tokenizer
              self.stop_str = stop_str.lower()
          def __call__(self, input_ids, scores, **kwargs):
              text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).lower()
              return self.stop_str in text

stops = StoppingCriteriaList([StopOnHTMLClose(tokenizer)])

# ========= 2️⃣ النص العربي =========
description = "A responsive website for an art gallery named ArtVerse that showcases modern and classic artworks. The color scheme uses DarkSlateGray for the header and navigation, White for background, LightGray for content sections, and Gold for interactive elements like buttons and highlights. The navigation bar includes: Home, Artists, Gallery, and Exhibitions. The website includes artistic images: an abstract painting, a classical portrait, and a digital art piece. The layout adjusts smoothly for small screens using media queries."


# ========= 3️⃣ ترجمة النص إلى الإنكليزية =========
#translated_text = GoogleTranslator(source='auto', target='en').translate(arabic_description)

#print("🔹 النص بعد الترجمة:")
#print(translated_text)

# ========= 4️⃣ توليد الكود =========
generation_config = GenerationConfig(
    #max_length=4096,
    max_new_tokens=2048,   # يكفي لموقع كامل HTML+CSS+JS متوسط
    num_beams=4,
    early_stopping=False,
    #early_stopping=True
    #do_sample=True,         # جرّب أيضًا num_beams=1 لو أردت بدون sampling
    #temperature=0.7,
    #top_p=0.95,
    #no_repeat_ngram_size=4,
    #repetition_penalty=1.1

)

inputs = tokenizer(
    description,
    return_tensors="pt",
    truncation=True,
    max_length=256,
    padding='max_length'
).to(device)

attention_mask = tokenizer(
    #"generate HTML: " + description,
    description,
    return_tensors="pt",
    truncation=True,
    max_length=2048,
    padding='max_length'
)['attention_mask'].to(device)

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"], generation_config=generation_config,stopping_criteria=stops)


generated_html = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

# ========= 5️⃣ حفظ الناتج =========
output_file = "generated_delete_replace.html"

with open(output_file, "w") as f:
    f.write(generated_html)

print(f"\n✅ تم حفظ الكود الناتج في الملف: {output_file}")
