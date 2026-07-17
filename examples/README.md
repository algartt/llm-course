# Çalıştırılabilir Framework Örnekleri

Bu klasörde aynı temel işi iki farklı ekosistemle yapan komut satırı örnekleri bulunur:

- `pytorch_text_generation.py`: PyTorch + Hugging Face Transformers.
- `tensorflow_keras_text_generation.py`: TensorFlow backend + KerasHub.

Örnekler ilk çalıştırmada model ağırlıklarını internetten indirir. Model lisansı veya erişim koşulu varsa ilgili platformda kabul etmeniz gerekebilir.

## PyTorch örneği

Sanal ortamı etkinleştirdikten sonra:

```bash
python -m pip install --upgrade -r examples/requirements-pytorch.txt
python examples/pytorch_text_generation.py
```

Özel prompt:

```bash
python examples/pytorch_text_generation.py \
  --prompt "RAG sistemini üç maddede açıkla." \
  --max-new-tokens 120 \
  --greedy
```

Windows PowerShell'de komutu tek satırda yazabilir veya satır devamı için backtick kullanabilirsiniz.

Betik otomatik olarak CUDA, Apple MPS veya CPU cihazını seçer ve üretilen token sayısıyla yaklaşık token/saniye değerini yazdırır.

## TensorFlow/Keras örneği

```bash
python -m pip install --upgrade -r examples/requirements-tensorflow.txt
python examples/tensorflow_keras_text_generation.py
```

Özel prompt ve sampling:

```bash
python examples/tensorflow_keras_text_generation.py \
  --prompt "LoRA yöntemini kısa biçimde açıkla." \
  --top-p 0.85 \
  --top-k 40
```

Betik Keras backend'ini importlardan önce TensorFlow olarak ayarlar. Varsayılan küçük preset `gemma3_instruct_270m` modelidir.

## Tekrarlanabilir ortam kaydı

Requirements dosyaları güncel uyumlu sürümleri kurmak için bilinçli olarak kesin sürüme sabitlenmemiştir. Başarılı bir denemeden sonra ortamı kaydedin:

```bash
python -m pip freeze > versions.txt
```

README veya deney raporuna ayrıca şunları yazın:

- Python sürümü,
- işletim sistemi,
- CPU/GPU modeli,
- model ve tokenizer kimliği,
- dtype/quantization,
- prompt ve sampling ayarları.

## Adil performans karşılaştırması

PyTorch ve TensorFlow/Keras sonuçlarını karşılaştırırken farklı model kullanmak framework hızını tek başına ölçmez. Adil benchmark için:

1. Aynı veya eşdeğer ağırlıkları kullanın.
2. Aynı prompt ve bağlam uzunluğunu koruyun.
3. Aynı dtype, batch ve üretim uzunluğunu ayarlayın.
4. İlk çağrıyı ısınma olarak ayırın.
5. En az 10 tekrar çalıştırın.
6. p50/p95 gecikme, throughput ve peak memory raporlayın.
7. Üretilen metnin kalitesinin bozulmadığını kontrol edin.

## Sorun giderme

| Hata | Çözüm |
|---|---|
| Model indirilemiyor | İnternet bağlantısını, disk alanını ve model erişim koşullarını kontrol edin. |
| CUDA out of memory | Daha küçük model seçin veya üretim uzunluğunu azaltın. |
| Keras backend yanlış | Python sürecini yeniden başlatın; `KERAS_BACKEND` importlardan önce ayarlanmalıdır. |
| Üretim boş/tekrarlı | Chat template, EOS/PAD ve sampling ayarlarını kontrol edin. |
| İlk çağrı çok yavaş | Model indirme/derleme/ısınma süresini sonraki çağrılardan ayrı ölçün. |

## Sonraki geliştirmeler

- Her iki betiğe batch prompt desteği ekleyin.
- Sonuçları JSONL dosyasına kaydedin.
- Aynı test promptlarını otomatik çalıştırın.
- Bellek ve p50/p95 gecikme ölçümü ekleyin.
- Sonuç tablosunu [framework karşılaştırma sayfasına](../guides/pytorch-vs-tensorflow.md) ekleyin.


