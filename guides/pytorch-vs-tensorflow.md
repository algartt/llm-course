# PyTorch mu TensorFlow/Keras mı?

İki framework de sinir ağı eğitimi ve çıkarımı için yeterince güçlüdür. En iyi seçim; ekip deneyimi, kullanılacak model ekosistemi, hedef donanım ve dağıtım ortamına bağlıdır.

## Kısa karşılaştırma

| Başlık | PyTorch | TensorFlow + Keras |
|---|---|---|
| Öğrenme deneyimi | Python'a yakın, açık eğitim döngüleri | Yüksek seviyeli `fit()` akışıyla hızlı başlangıç |
| LLM ekosistemi | Açık kaynak LLM araştırma ve ince ayar örneklerinde çok yaygın | KerasHub ile backend bağımsız yüksek seviye model API'leri |
| Özelleştirme | Dinamik kontrol akışı ve düşük seviyeli müdahale güçlü | Keras katman/model soyutlamalarıyla düzenli yapı |
| Veri hattı | `Dataset` ve `DataLoader` | `tf.data` |
| Dağıtım | TorchScript/Export, ONNX ve çeşitli sunucular | SavedModel ekosistemi, TFLite ve TensorFlow Serving |
| Mobil/edge | ExecuTorch ve üçüncü taraf yollar | TensorFlow Lite güçlü bir seçenek |
| Çoklu backend | Ana yürütme PyTorch'tur | Keras 3 TensorFlow, JAX ve PyTorch backend destekler |
| En uygun başlangıç | Transformers/PEFT ağırlıklı LLM projesi | Mevcut TensorFlow altyapısı veya KerasHub projesi |

Bu tablo mutlak bir performans sıralaması değildir. Aynı modelin hızı; kernel, dtype, batch, bağlam uzunluğu, derleme, donanım ve sürüme göre değişir.

## Seçim ağacı

- Kullanacağınız açık kaynak LLM örneği yalnızca PyTorch/Transformers ile destekleniyorsa **PyTorch** ile başlayın.
- Ekibiniz TensorFlow üretim altyapısına sahipse veya TFLite hedefliyorsa **TensorFlow/Keras** daha düşük entegrasyon maliyeti sağlayabilir.
- Modeli KerasHub üzerinden kullanacak ve backend değiştirme esnekliği istiyorsanız **Keras 3** değerlendirin.
- Araştırma koduna sık sık düşük seviyeli müdahale edecekseniz **PyTorch** genellikle daha doğrudan bir deneyim sunar.
- Karar hâlâ net değilse, küçük bir benchmark hazırlayıp gerçek veri ve hedef donanım üzerinde ölçün.

## Adil benchmark nasıl yapılır?

Aynı koşulları koruyun:

1. Aynı model boyutu ve mümkünse aynı ağırlıklar.
2. Aynı tokenizer, prompt ve bağlam uzunluğu.
3. Aynı dtype ve quantization düzeyi.
4. Aynı batch boyutu ve üretilecek token sayısı.
5. Isınma çağrılarından sonra en az birkaç tekrar.
6. Hem gecikme hem throughput ölçümü.
7. En yüksek GPU/CPU belleği ve çıktı kalitesi kontrolü.

Basit rapor şablonu:

| Metrik | PyTorch | TensorFlow/Keras |
|---|---:|---:|
| İlk çağrı süresi |  |  |
| Isınmış p50 gecikme |  |  |
| Isınmış p95 gecikme |  |  |
| Token/saniye |  |  |
| En yüksek bellek |  |  |
| Çıktı kalite skoru |  |  |

## Framework'ten bağımsız doğru alışkanlıklar

- Veriyi train/validation/test olarak ayırın ve sızıntıyı kontrol edin.
- Ön işlem ile tokenizer sürümünü modelle birlikte kaydedin.
- Seed, paket sürümleri, donanım ve hiperparametreleri raporlayın.
- Baseline olmadan karmaşık model denemelerine başlamayın.
- Ortalama skorların yanında hatalı örnekleri inceleyin.
- Model kalitesini maliyet, gecikme, bellek ve güvenlikle birlikte değerlendirin.
- Gizli anahtarları `.env` veya secret manager içinde tutun; Git geçmişine eklemeyin.

## Aynı proje için iki uygulama fikri

Frameworkleri gerçekten karşılaştırmak için aynı Türkçe duygu analizi projesini iki kez uygulayın:

1. Veri temizleme ve bölme kodunu ortak tutun.
2. PyTorch tarafında `Dataset/DataLoader`, TensorFlow tarafında `tf.data` kullanın.
3. Benzer boyutta embedding ve sınıflandırma katmanları seçin.
4. Accuracy, macro-F1, eğitim süresi ve peak memory kaydedin.
5. Her iki modeli aynı API sözleşmesinin arkasında sunun.
6. Sonuçları README'de tablo ve başarısız örneklerle açıklayın.

Bu tür bir proje, yalnızca çalışan kod değil; deney tasarımı, ölçüm ve teknik iletişim becerisi de gösterir.

## İlgili rehberler

- [PyTorch ile LLM Uygulamaları](pytorch-llm.md)
- [TensorFlow ve Keras ile LLM Uygulamaları](tensorflow-keras-llm.md)
- [PyTorch resmi belgeleri](https://docs.pytorch.org/docs/stable/index.html)
- [TensorFlow resmi belgeleri](https://www.tensorflow.org/guide)
- [Keras resmi belgeleri](https://keras.io/)

