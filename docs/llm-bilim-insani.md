# LLM Bilim İnsanı Rotası

Bu rota model mimarisi, veri hazırlama, ince ayar, tercih hizalama, değerlendirme ve kuantizasyon konularına odaklanır. Amaç yalnızca bir eğitim betiğini çalıştırmak değil, deneyin neden başarılı veya başarısız olduğunu ölçebilmektir.

## Öğrenme hedefleri

- Decoder-only Transformer veri akışını açıklamak.
- Eğitim sonrası veri kümesi ve chat template hazırlamak.
- Tam ince ayar ile LoRA/QLoRA arasındaki farkı değerlendirmek.
- Model kalitesini baseline ve hata analiziyle ölçmek.
- Deneyleri tekrarlanabilir biçimde raporlamak.

## 1. Transformer mimarisi

Modern üretken LLM'lerin çoğu decoder-only yapıyı kullanır. Basitleştirilmiş akış:

```text
Metin
  ↓ tokenizer
Token kimlikleri
  ↓ embedding + konumsal bilgi
Transformer blokları
  ↓ normalization
Kelime dağarcığı logitleri
  ↓ sampling
Yeni token
```

Bir Transformer bloğunda genellikle:

- self-attention,
- çok başlı attention,
- feed-forward/MLP katmanı,
- normalization,
- residual bağlantılar bulunur.

### Bilmeniz gereken mimari seçimler

- **RoPE:** Token konum bilgisini attention hesabına taşır.
- **GQA/MQA:** KV cache maliyetini azaltmak için key/value head sayısını düşürür.
- **RMSNorm:** Layer normalization'ın yaygın bir alternatifidir.
- **SwiGLU/GeGLU:** MLP katmanlarında kullanılan kapılı aktivasyonlardır.
- **MoE:** Her token için yalnızca seçilen uzman ağlarını çalıştırır.

## 2. Ön eğitim

Ön eğitim büyük miktarda ham metinden sonraki token tahmini öğrenir. Bireysel projelerde sıfırdan büyük model eğitmek çoğunlukla gereksizdir; fakat süreci anlamak önemlidir.

### Veri hattı

1. Kaynakları ve kullanım izinlerini belirleme.
2. Dil ve kalite filtreleme.
3. Kişisel/gizli veri temizleme.
4. Tam ve yakın kopyaları kaldırma.
5. Test verisiyle çakışmayı kontrol etme.
6. Tokenizasyon ve sequence packing.
7. Veri karışımı oranlarını kaydetme.

Veri kalitesi, yalnızca veri miktarından daha önemlidir. Kötü biçimlendirilmiş, tekrarlı veya lisansı belirsiz veri sonraki aşamalarda ciddi sorun oluşturur.

### Eğitim izlemesi

- Train/validation loss.
- Learning rate ve gradient norm.
- Token/saniye ve GPU kullanımı.
- Hatalı veya boş batch sayısı.
- Checkpoint ve veri konumu.
- NaN/Inf kontrolleri.

## 3. Eğitim sonrası veri

Talimat verisi çoğunlukla mesaj dizileriyle tutulur:

```json
{
  "messages": [
    {"role": "system", "content": "Kısa ve doğru cevap ver."},
    {"role": "user", "content": "RAG nedir?"},
    {"role": "assistant", "content": "RAG, üretimi getirilen bağlamla destekler."}
  ]
}
```

Kalite kontrolleri:

- Zorunlu alanlar ve geçerli roller.
- Boş veya aşırı uzun mesajlar.
- Yinelenen örnekler.
- Cevap ile talimat uyumu.
- Dil, konu ve zorluk dengesi.
- PII, güvenlik ve lisans kontrolleri.
- Chat template sonundaki EOS davranışı.

## 4. Gözetimli ince ayar

### Tam ince ayar

Bütün model ağırlıkları güncellenir. Esnektir fakat yüksek GPU belleği, depolama ve eğitim süresi gerektirir.

### LoRA

Temel ağırlıklar dondurulur; attention gibi belirli doğrusal katmanlara düşük rank adaptörleri eklenir.

Önemli parametreler:

- `r`: Adaptör rank değeri.
- `lora_alpha`: LoRA güncellemesinin ölçeği.
- `lora_dropout`: Düzenlileştirme.
- `target_modules`: Adaptörün ekleneceği katmanlar.

### QLoRA

Temel model düşük bitli biçimde tutulurken LoRA adaptörleri daha yüksek hassasiyetle eğitilir. Belleği ciddi biçimde düşürebilir; hız kazancı donanıma bağlıdır.

### Deney tasarımı

İlk deneyde yalnızca bir değişkeni değiştirin:

| Deney | Veri | Rank | Learning rate | Epoch | Amaç |
|---|---:|---:|---:|---:|---|
| Baseline | 1k | 8 | 2e-4 | 1 | Boru hattı kontrolü |
| A | 1k | 16 | 2e-4 | 1 | Rank etkisi |
| B | 1k | 8 | 1e-4 | 1 | Öğrenme oranı etkisi |
| C | 5k | 8 | 2e-4 | 1 | Veri miktarı etkisi |

## 5. Tercih hizalama

Tercih veri kümesi aynı talimat için seçilen ve reddedilen cevapları içerir:

```json
{
  "prompt": "Güvenli parola nasıl oluşturulur?",
  "chosen": "Uzun, benzersiz ve parola yöneticisinde saklanan bir parola kullanın.",
  "rejected": "Her yerde aynı kısa parolayı kullanın."
}
```

- **RLHF:** Ödül modeli ve reinforcement learning aşamaları içerir.
- **DPO:** Seçilen cevabı reddedilene göre doğrudan tercih edecek şekilde optimize eder.
- **ORPO/benzeri yöntemler:** Gözetimli hedef ile tercih sinyalini farklı biçimlerde birleştirir.

Tercih verisindeki tutarsız etiketler, yöntemin seçilmesinden daha büyük etki yaratabilir. İnsan etiketleyiciler için açık ölçütler hazırlayın.

## 6. Değerlendirme

### Üç katmanlı değerlendirme

1. **Görev başarısı:** Doğru cevap, format uyumu veya otomatik metrik.
2. **Kalite:** Tutarlılık, açıklık, kaynak kullanımı ve dil kalitesi.
3. **Risk:** Zararlı içerik, veri sızıntısı, önyargı ve prompt saldırıları.

### LLM-as-a-judge

Hakem model hızlı ve ölçeklenebilir olabilir; fakat konum yanlılığı, uzun cevap tercihi ve kendine benzer modeli kayırma gibi sorunlar taşır. Önlemler:

- Cevap sırasını değiştirerek iki kez değerlendirme.
- Açık rubric ve örnekler kullanma.
- İnsan etiketli küçük bir alt kümeyle kalibrasyon.
- Hakem model/prompt sürümünü kaydetme.
- Yalnızca toplam skoru değil gerekçeleri de inceleme.

### Veri sızıntısı

Değerlendirme soruları eğitim verisinde bulunuyorsa skor gerçeği yansıtmaz. N-gram, embedding benzerliği ve kaynak metaverisiyle çakışma kontrolü yapın.

## 7. Kuantizasyon

Kuantizasyon, ağırlık veya aktivasyonları daha düşük bit hassasiyetinde temsil eder.

| Yöntem | Genel kullanım |
|---|---|
| FP16/BF16 | GPU çıkarımı ve eğitim |
| INT8 | Bellek azaltma, görece düşük kalite kaybı |
| 4-bit | Büyük modeli sınırlı bellekte çalıştırma |
| GGUF | llama.cpp ve yerel CPU/GPU çıkarımı |
| GPTQ/AWQ | GPU ağırlıklı düşük bit çıkarımı |

Kuantize modeli her zaman hedef görevde yeniden değerlendirin. Perplexity değişimi tek başına kullanıcı deneyimini açıklamaz.

## 8. Deney kayıt şablonu

```yaml
experiment: tr-sft-lora-001
base_model: model-kimligi
dataset: veri-kumesi-ve-surumu
seed: 42
sequence_length: 512
train_examples: 5000
method: lora
lora_rank: 8
learning_rate: 0.0002
epochs: 1
hardware: gpu-modeli
framework_versions: versions.txt
```

Kaydedilecek çıktılar:

- yapılandırma,
- eğitim ve doğrulama metrikleri,
- checkpoint/adaptör,
- sabit değerlendirme promptları,
- başarısız örnekler,
- model kartı ve lisans bilgisi.

## 9. Portföy laboratuvarı

### Proje: Türkçe talimat adaptörü

1. Lisansı uygun küçük bir instruction model seçin.
2. 500–5.000 kaliteli Türkçe örnek hazırlayın.
3. Train/validation/test ayrımı yapın.
4. Temel model baseline sonuçlarını kaydedin.
5. LoRA ile kısa bir eğitim gerçekleştirin.
6. Otomatik rubric ve insan incelemesini birleştirin.
7. Bellek, süre ve kalite değişimini tabloyla raporlayın.
8. Adaptör, örnek kod ve model kartını yayımlayın.

## Tamamlama ölçütü

- [ ] Transformer veri akışını tensör boyutlarıyla açıklayabiliyorum.
- [ ] Chat template ve EOS davranışını test ettim.
- [ ] Baseline ile ince ayarlı modeli aynı test kümesinde karşılaştırdım.
- [ ] Deney yapılandırmasını ve paket sürümlerini kaydettim.
- [ ] Başarısız örnekler ve güvenlik riskleri için bölüm yazdım.

## Sonraki adım

- [LLM Mühendisi Rotası](llm-muhendisi.md)
- [PyTorch ile LLM Uygulamaları](../guides/pytorch-llm.md)
- [TensorFlow ve Keras ile LLM Uygulamaları](../guides/tensorflow-keras-llm.md)


