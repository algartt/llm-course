# TensorFlow ve Keras ile LLM Uygulamaları

Bu rehber TensorFlow arka ucuyla Keras 3 ve KerasHub kullanarak dil modeli çıkarımı, veri hattı ve düşük maliyetli ince ayar için başlangıç sağlar.

## 1. Kurulum

```bash
python -m pip install tensorflow keras keras-hub
```

NVIDIA GPU kurulumu işletim sistemi ve sürücüye bağlıdır. Kurulumdan sonra cihazları kontrol edin:

```python
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPU'lar:", tf.config.list_physical_devices("GPU"))
```

## 2. Keras arka ucunu ayarlama

Keras 3 birden fazla backend destekler. TensorFlow kullanmak için ortam değişkenini `keras` veya `keras_hub` içe aktarılmadan önce ayarlayın:

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import keras_hub
import tensorflow as tf

print("Backend:", keras.backend.backend())
```

Bu satırı importlardan sonra çalıştırmak mevcut Python sürecinin backend'ini güvenilir biçimde değiştirmez. Notebook çekirdeğini yeniden başlatmanız gerekebilir.

## 3. KerasHub ile metin üretimi

KerasHub `CausalLM.from_preset()` ile ön işleyici, omurga ve dil modeli başlığını birlikte yükleyebilir:

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_hub

model = keras_hub.models.CausalLM.from_preset(
    "gemma2_instruct_2b_en",
    dtype="bfloat16",
)

prompt = "Büyük dil modellerinde attention mekanizmasını kısa biçimde açıkla."
response = model.generate(prompt, max_length=160)
print(response)
```

Preset ilk kullanımda model dosyalarını indirir. Bazı modeller erişim onayı, kimlik doğrulama veya yüksek bellek gerektirebilir. Donanımınız `bfloat16` için uygun değilse `dtype` parametresini kaldırarak varsayılan türle başlayın.

## 4. Sampling ayarları

KerasHub üretim stratejisini `compile()` üzerinden değiştirebilir:

```python
sampler = keras_hub.samplers.TopPSampler(
    p=0.9,
    k=50,
    seed=42,
)

model.compile(sampler=sampler)
print(model.generate(prompt, max_length=160))
```

- Düşük çeşitlilik gereken bilgi çıkarma işlerinde greedy yaklaşım uygundur.
- Yaratıcı üretimde temperature/top-p kullanılabilir.
- Karşılaştırmalı değerlendirmede seed ve üretim ayarlarını kaydedin.
- `max_length` toplam dizi uzunluğunu etkileyebilir; prompt uzunluğunu hesaba katın.

## 5. TensorFlow veri hattı

`tf.data` ile metinleri karıştırıp batch ve prefetch uygulayabilirsiniz:

```python
import tensorflow as tf

train_texts = [
    "Soru: Token nedir?\nCevap: Bir modelin metni işlerken kullandığı temel parçadır.",
    "Soru: RAG nedir?\nCevap: Üretimi dış kaynaklardan getirilen bağlamla destekleyen yaklaşımdır.",
    "Soru: LoRA nedir?\nCevap: Az sayıda ek parametreyle verimli ince ayar yöntemidir.",
]

dataset = (
    tf.data.Dataset.from_tensor_slices(train_texts)
    .shuffle(buffer_size=len(train_texts), seed=42)
    .batch(1)
    .prefetch(tf.data.AUTOTUNE)
)
```

Büyük veri kümelerinde bütün metni belleğe yüklemeyin. Dosyadan akış, `cache()` davranışı, deterministik karıştırma ve train/validation ayrımı için veri hattını açıkça tasarlayın.

## 6. LoRA ile ince ayar

KerasHub omurgası destekliyorsa LoRA etkinleştirilebilir:

```python
model.backbone.enable_lora(rank=4)
model.preprocessor.sequence_length = 256

model.fit(dataset, epochs=1)
```

`CausalLM` sınıfı eğitim için gerekli kaybı varsayılan olarak yapılandırır. Özel optimizer veya `compile()` ayarı kullanıyorsanız veri biçiminin, maskelerin ve etiket kaydırmasının ilgili model API'siyle uyumunu doğrulayın. İlk denemeyi birkaç örnek üzerinde yapıp loss değerinin mantıklı olduğunu kontrol edin.

Eğitimden önce:

```python
from math import prod

trainable = sum(prod(v.shape) for v in model.trainable_weights)
total = sum(prod(v.shape) for v in model.weights)

print(f"Eğitilebilir parametre: {trainable:,}")
print(f"Toplam parametre: {total:,}")
print(f"Eğitilebilir oran: %{100 * trainable / total:.2f}")
```

Şekil boyutlarından biri tanımsızsa sayımı eğitim verisiyle ilk çağrıdan sonra yapın. `model.summary()` ve değişkenlerin `trainable` durumları da LoRA'nın etkisini görmenizi sağlar.

## 7. Klasik Keras metin sınıflandırma örneği

LLM dışındaki metin görevlerinde daha küçük bir model çoğu zaman daha hızlı ve ucuzdur:

```python
import keras

max_tokens = 20_000
sequence_length = 256

vectorizer = keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# Yalnızca eğitim metinleriyle sözlük oluşturun.
vectorizer.adapt(train_text_dataset)

inputs = keras.Input(shape=(1,), dtype="string")
x = vectorizer(inputs)
x = keras.layers.Embedding(max_tokens, 128, mask_zero=True)(x)
x = keras.layers.Bidirectional(keras.layers.LSTM(64))(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

classifier = keras.Model(inputs, outputs)
classifier.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
```

`vectorizer.adapt()` çağrısını test verisi üzerinde çalıştırmak veri sızıntısına yol açar. Sözlüğü yalnızca eğitim bölümüyle oluşturun.

## 8. Kaydetme ve yeniden yükleme

Keras modeli:

```python
classifier.save("artifacts/text_classifier.keras")
restored = keras.models.load_model("artifacts/text_classifier.keras")
```

KerasHub preset/model kaydetme seçenekleri kullanılan sınıfa göre değişebilir. LoRA adaptörü, tokenizer/preprocessor, model preset kimliği, backend ve paket sürümlerini birlikte kaydedin. Yalnızca ağırlıkları saklamak üretim ortamında aynı ön işlemenin kurulacağını garanti etmez.

## 9. Performans önerileri

- `tf.data` hattında `prefetch(tf.data.AUTOTUNE)` kullanın.
- Desteklenen hızlandırıcılarda mixed precision deneyin ve sayısal kararlılığı ölçün.
- Benzer uzunluktaki girdileri aynı batch'e gruplayın.
- XLA/JIT kullanımını gerçek iş yükü üzerinde benchmark edin.
- Uzun bağlam ve büyük batch değerlerini birlikte artırmayın.
- Modeli ısındırdıktan sonra gecikme ve throughput ölçün.

Mixed precision örneği:

```python
from keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")
```

Bu politika her donanımda hız kazandırmaz. Çıktı katmanı ve özel kayıplarda dtype davranışını kontrol edin.

## 10. Sık karşılaşılan sorunlar

| Sorun | Kontrol |
|---|---|
| Backend yanlış görünüyor | `KERAS_BACKEND` değişkenini importlardan önce ayarlayın ve süreci yeniden başlatın. |
| GPU görünmüyor | TensorFlow, sürücü ve işletim sistemi uyumluluğunu resmi kurulum matrisinden kontrol edin. |
| Bellek yetmiyor | Küçük preset, kısa `sequence_length`, batch=1 ve LoRA kullanın. |
| Üretim çok yavaş | İlk çağrının derleme/ısınma maliyetini ayırın; sonraki çağrıları ayrıca ölçün. |
| Eğitim biçimi uyuşmuyor | Preprocessor çıktısını, padding/mask davranışını ve hedef şekillerini tek batch üzerinde yazdırın. |

## Resmi kaynaklar

- [TensorFlow kurulum belgeleri](https://www.tensorflow.org/install)
- [Keras 3 başlangıç rehberi](https://keras.io/getting_started/)
- [KerasHub başlangıç rehberi](https://keras.io/keras_hub/getting_started/)
- [KerasHub CausalLM API](https://keras.io/keras_hub/api/base_classes/causal_lm/)
- [KerasHub model listesi](https://keras.io/keras_hub/api/models/)

