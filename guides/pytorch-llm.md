# PyTorch ile LLM Uygulamaları

Bu rehber, PyTorch ve Hugging Face Transformers kullanarak küçük bir dil modelini çalıştırmayı, bir sınıflandırma modelini eğitmeyi ve daha büyük projelere geçerken dikkat edilmesi gereken noktaları gösterir.

## 1. Kurulum

Önce PyTorch'u donanımınıza uygun komutla [resmi kurulum sayfasından](https://pytorch.org/get-started/locally/) kurun. Ardından:

```bash
python -m pip install transformers datasets evaluate accelerate scikit-learn
```

Kurulumu kontrol edin:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA kullanılabilir:", torch.cuda.is_available())
print("MPS kullanılabilir:", torch.backends.mps.is_available())
```

## 2. Cihaz seçimi

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Cihaz:", device)
```

CUDA, NVIDIA GPU'ları; MPS ise desteklenen Apple Silicon cihazları kullanır. CPU ile de çalışabilirsiniz ancak üretim ve eğitim daha yavaş olur.

## 3. Hazır modelle Türkçe metin üretimi

Aşağıdaki örnek küçük bir instruction modelini yükler. İlk çalıştırmada model dosyaları indirilir.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.to(device)
model.eval()

messages = [
    {
        "role": "user",
        "content": "Büyük dil modellerinde tokenizasyonu kısa ve anlaşılır biçimde açıkla.",
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

### Üretim ayarlarını nasıl okumalı?

- `max_new_tokens`: Üretilecek en fazla yeni token sayısıdır.
- `temperature`: Düşük değer daha tutarlı, yüksek değer daha çeşitli çıktı üretir.
- `top_p`: Olasılık kütlesinin belirli bölümündeki tokenları örnekler.
- `do_sample=False`: Daha deterministik greedy üretime geçer.
- `repetition_penalty`: Aşırı tekrarları sınırlamaya yardım eder.

Kaliteyi tek bir örnekle değerlendirmeyin. Sabit bir test prompt kümesi hazırlayıp doğruluk, tutarlılık, gecikme ve başarısızlık türlerini birlikte kaydedin.

## 4. Batch üretimi

Birden fazla girdiyi aynı anda çalıştırmak GPU kullanımını iyileştirebilir:

```python
prompts = [
    "Transformer mimarisini bir paragrafta özetle.",
    "RAG sisteminin üç temel bileşenini say.",
]

tokenizer.padding_side = "left"
batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

with torch.inference_mode():
    outputs = model.generate(
        **batch,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

for prompt_ids, output in zip(batch["input_ids"], outputs):
    generated = output[prompt_ids.shape[0]:]
    print(tokenizer.decode(generated, skip_special_tokens=True))
```

Padding içeren gerçek projelerde giriş uzunluklarını attention mask ile takip edin. Benzer uzunluktaki örnekleri aynı batch'e koymak gereksiz hesaplamayı azaltır.

## 5. Metin sınıflandırma modeli eğitimi

Bu bölüm IMDB veri kümesinin küçük bir bölümünü kullanır. Resmi Transformers akışı `AutoModelForSequenceClassification`, `TrainingArguments` ve `Trainer` sınıflarına dayanır.

```python
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

checkpoint = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

dataset = load_dataset("imdb")
train_ds = dataset["train"].shuffle(seed=42).select(range(2000))
test_ds = dataset["test"].shuffle(seed=42).select(range(500))

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
)

args = TrainingArguments(
    output_dir="outputs/imdb-distilbert",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
print(trainer.evaluate())
trainer.save_model("outputs/imdb-distilbert/best")
tokenizer.save_pretrained("outputs/imdb-distilbert/best")
```

Türkçe sınıflandırma için Türkçe veya çok dilli bir checkpoint ve lisansı uygun Türkçe veri kümesi seçin. Sınıf dağılımı dengesizse accuracy yanında precision, recall ve F1 raporlayın.

## 6. LoRA ile verimli ince ayar yaklaşımı

Tam ince ayar bütün ağırlıkları günceller ve yüksek bellek gerektirir. LoRA, seçilen katmanlara küçük eğitilebilir matrisler ekleyerek maliyeti düşürür. Tipik akış:

1. Uygun lisanslı bir temel/instruction model seçin.
2. Sohbet şablonunu ve EOS token davranışını doğrulayın.
3. `peft` ile LoRA hedef katmanlarını tanımlayın.
4. Eğitim verisini `trl` veya özel eğitim döngüsüyle kullanın.
5. Temel model ile adaptörü ayrı kaydedin; gerekiyorsa sonradan birleştirin.

```bash
python -m pip install peft trl
```

```python
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
```

Katman adları model mimarisine göre değişir. `target_modules` listesini kopyalamadan önce `model.named_modules()` çıktısıyla doğrulayın.

## 7. Performans ve bellek

- Çıkarımda `model.eval()` ve `torch.inference_mode()` kullanın.
- Desteklenen GPU'larda `float16` veya `bfloat16` belleği azaltabilir.
- Eğitimde gradient accumulation küçük batch sorununu hafifletir.
- Gradient checkpointing belleği azaltır, fakat eğitimi yavaşlatabilir.
- Uzun bağlam, attention maliyetini hızla artırır; gereksiz tokenları ayıklayın.
- Kuantizasyon kalite, bellek ve hız arasında donanıma bağlı bir dengedir.

Örnek otomatik dtype ve cihaz yerleşimi:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
```

`device_map="auto"` için `accelerate` gerekir. Modelin bazı katmanları CPU'ya taşınırsa gecikme belirgin biçimde artabilir.

## 8. Sık karşılaşılan sorunlar

| Sorun | Kontrol |
|---|---|
| CUDA out of memory | Batch, bağlam uzunluğu ve `max_new_tokens` değerini küçültün; daha küçük/kuantize model deneyin. |
| Anlamsız veya tekrarlı çıktı | Sohbet şablonunu, EOS/PAD tokenlarını ve sampling ayarlarını doğrulayın. |
| Eğitim kaybı düşmüyor | Etiketleri, attention mask'i, öğrenme oranını ve veri biçimini küçük örnek üzerinde inceleyin. |
| Sonuçlar değişiyor | Seed belirleyin; veri sırası, sampling ve donanım kaynaklı deterministik olmayan işlemleri kaydedin. |
| Model indirilemiyor | Model erişim koşullarını kabul edin, kimlik doğrulamayı ve disk alanını kontrol edin. |

## 9. Mini proje kontrol listesi

- [ ] Model ve veri lisansını kaydettim.
- [ ] Eğitim/test ayrımında veri sızıntısını kontrol ettim.
- [ ] Baseline sonuç ürettim.
- [ ] Seed, paket sürümü ve hiperparametreleri kaydettim.
- [ ] Başarısız örnekleri sınıflandırdım.
- [ ] Bellek, gecikme ve kalite ölçtüm.
- [ ] Model kartına kullanım sınırlarını yazdım.

## Resmi kaynaklar

- [PyTorch belgeleri](https://docs.pytorch.org/docs/stable/index.html)
- [Transformers metin üretimi belgeleri](https://huggingface.co/docs/transformers/main/main_classes/text_generation)
- [Transformers metin sınıflandırma rehberi](https://huggingface.co/docs/transformers/main/tasks/sequence_classification)
- [PEFT LoRA belgeleri](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)


