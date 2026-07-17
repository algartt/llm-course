<div align="center">
  <img src="img/banner.png" alt="LLM Course">
  <h1>LLM Kursu — Türkçe Yol Haritası</h1>
  <p>Büyük dil modellerini temelden üretime kadar öğrenmek için düzenlenmiş Türkçe kaynak ve uygulama rehberi.</p>
</div>

> Bu depo, [mlabonne/llm-course](https://github.com/mlabonne/llm-course) projesinin Türkçe yerelleştirmesidir. Özgün İngilizce metin [README_EN.md](README_EN.md) dosyasında korunur.

## Bu depoda neler var?

Kurs üç ana bölümden oluşur:

1. **LLM Temelleri:** Matematik, Python, sinir ağları ve doğal dil işleme.
2. **LLM Bilim İnsanı:** Transformer mimarisi, ön eğitim, ince ayar, tercih hizalama, değerlendirme ve kuantizasyon.
3. **LLM Mühendisi:** Model çalıştırma, RAG, vektör veritabanları, ajanlar, çıkarım optimizasyonu, dağıtım ve güvenlik.

Türkçe sürüme ayrıca uygulamalı framework rehberleri eklendi:

| Rehber | Kapsam |
|---|---|
| [PyTorch ile LLM Uygulamaları](guides/pytorch-llm.md) | Kurulum, metin üretimi, eğitim, LoRA yaklaşımı, performans ve hata ayıklama |
| [TensorFlow ve Keras ile LLM Uygulamaları](guides/tensorflow-keras-llm.md) | KerasHub, TensorFlow veri hattı, üretim, LoRA ve model kaydetme |
| [PyTorch mu TensorFlow mu?](guides/pytorch-vs-tensorflow.md) | Framework karşılaştırması, seçim ağacı ve ortak proje planı |

## Hızlı başlangıç

Python 3.10 veya daha yeni bir sürüm ve mümkünse GPU kullanın. Her proje için ayrı sanal ortam oluşturun:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Ardından seçtiğiniz rehberdeki paketleri kurun. Büyük modeller ciddi miktarda bellek kullanır; ilk denemelerde küçük modeller, kısa bağlam ve düşük batch boyutu tercih edin.

## Önerilen öğrenme sırası

### 1. Temel katman

- **Lineer cebir:** Vektörler, matrisler, matris çarpımı ve özdeğer sezgisi.
- **Türev ve optimizasyon:** Gradyan, zincir kuralı, gradyan inişi ve Adam.
- **Olasılık:** Dağılımlar, beklenen değer, varyans ve maksimum olabilirlik.
- **Python veri araçları:** NumPy, pandas, Matplotlib ve scikit-learn.
- **Sinir ağları:** Katman, aktivasyon, kayıp fonksiyonu, geri yayılım ve düzenlileştirme.
- **NLP:** Tokenizasyon, gömmeler, dikkat mekanizması ve dizi modelleme.

### 2. LLM bilim insanı rotası

1. **Transformer mimarisi:** Self-attention, çok başlı dikkat, konumsal bilgi, normalizasyon ve artık bağlantılar.
2. **Ön eğitim:** Veri toplama, temizleme, tokenizasyon, nedensel dil modelleme ve ölçekleme yasaları.
3. **Eğitim sonrası veri:** Talimat verisi hazırlama, kalite kontrolü, şablonlar ve veri karışımları.
4. **Gözetimli ince ayar:** Tam ince ayar, LoRA/QLoRA ve hiperparametre seçimi.
5. **Tercih hizalama:** Ödül modelleme, RLHF, DPO ve benzeri doğrudan tercih yöntemleri.
6. **Değerlendirme:** Görev metrikleri, insan değerlendirmesi, LLM-as-a-judge ve veri sızıntısı kontrolleri.
7. **Kuantizasyon:** 8-bit/4-bit ağırlıklar, GGUF, GPTQ, AWQ ve donanım dengeleri.
8. **Yeni eğilimler:** Uzun bağlam, multimodal modeller, seyrek uzman modelleri ve sentetik veri.

### 3. LLM mühendisi rotası

1. **Model çalıştırma:** Prompt şablonları, sampling, batch üretimi ve bağlam yönetimi.
2. **Vektör depolama:** Chunking, embedding, benzerlik araması ve metaveri filtreleri.
3. **RAG:** Sorgu, getirme, yeniden sıralama, bağlam oluşturma ve kaynak gösterme.
4. **Gelişmiş RAG:** Hibrit arama, sorgu dönüşümü, değerlendirme ve önbellekleme.
5. **Ajanlar:** Araç kullanımı, durum yönetimi, planlama ve güvenli yürütme sınırları.
6. **Çıkarım optimizasyonu:** KV cache, quantization, speculative decoding ve sürekli batch.
7. **Dağıtım:** API, container, gözlemlenebilirlik, maliyet ve ölçeklendirme.
8. **Güvenlik:** Prompt injection, veri sızıntısı, yetkilendirme, çıktı doğrulama ve kırmızı takım testleri.

## Uygulamalı notebook seçkisi

Özgün kursun Colab çalışmalarından öne çıkanlar:

| Konu | Ne öğretir? | Kaynak |
|---|---|---|
| Llama 3.1 + Unsloth | Verimli gözetimli ince ayar | [Notebook](https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z?usp=sharing) |
| Mistral-7B + QLoRA | Sınırlı GPU belleğiyle ince ayar | [Notebook](https://colab.research.google.com/drive/1o_w0KastmEJNVwT5GoqMCciH-18ca5WS?usp=sharing) |
| Mistral-7B + DPO | Tercih verisiyle hizalama | [Notebook](https://colab.research.google.com/drive/15iFBr1xWgztXvhrj5I9fBv20c7CFOPBE?usp=sharing) |
| 4-bit GPTQ | Tüketici donanımı için kuantizasyon | [Notebook](https://colab.research.google.com/drive/1lSvVDaRgqQp_mWK_jC9gydz6_-y6Aq4A?usp=sharing) |
| GGUF + llama.cpp | Yerel çıkarım için model hazırlama | [Notebook](https://colab.research.google.com/drive/1pL8k7m04mgE5jo2NrjGi8atB0j_37aDD?usp=sharing) |
| MergeKit | Modelleri yeniden eğitim olmadan birleştirme | [Notebook](https://colab.research.google.com/drive/1_JS7JKJAQozD48-LhYdegcuuZ2ddgXfr?usp=sharing) |
| Decoding stratejileri | Beam search, temperature ve nucleus sampling | [Notebook](https://colab.research.google.com/drive/19CJlOS5lI29g-B3dziNn93Enez1yiHk2?usp=sharing) |

> Notebook paket sürümleri zamanla değişebilir. Bir çalışma bozulursa sürümleri sabitleyin, hata mesajını okuyun ve ilgili kütüphanenin güncel resmi dokümantasyonunu kontrol edin.

## 30 günlük çalışma planı

| Günler | Hedef | Çıktı |
|---|---|---|
| 1–5 | Python, tensörler, veri kümeleri ve temel sinir ağları | Küçük bir metin sınıflandırıcı |
| 6–10 | Tokenizasyon, embedding ve attention | Attention boyutlarını açıklayan notebook |
| 11–15 | Hazır bir LLM ile çıkarım ve prompt denemeleri | Tekrarlanabilir üretim betiği |
| 16–20 | Küçük veri kümesiyle ince ayar veya LoRA | Eğitim günlüğü ve model çıktıları |
| 21–25 | Embedding tabanlı arama ve basit RAG | Kaynak gösteren soru-cevap demosu |
| 26–30 | Değerlendirme, API ve güvenlik kontrolleri | README, test örnekleri ve çalışan demo |

## Örnek portföy projeleri

- **Türkçe belge asistanı:** PDF/metin parçalama, embedding, kaynaklı cevap ve RAG değerlendirmesi.
- **Duygu analizi servisi:** PyTorch veya TensorFlow ile eğitim, REST API ve hata analizi.
- **Yerel sohbet arayüzü:** Küçük, kuantize bir model; token akışı ve oturum geçmişi.
- **Prompt değerlendirme paneli:** Aynı veri kümesinde farklı prompt ve sampling ayarlarının karşılaştırılması.
- **Model maliyet karşılaştırması:** Gecikme, throughput, bellek ve kalite metriklerinin raporlanması.

## Üretime geçmeden önce

- Eğitim, doğrulama ve test verilerini birbirinden ayırın.
- Rastgelelik kaynaklarını ve paket sürümlerini kaydedin.
- Yalnızca ortalama skora değil, başarısız örneklere de bakın.
- Model ve veri lisanslarını dağıtımdan önce kontrol edin.
- Gizli bilgileri promptlara, loglara veya depoya yazmayın.
- RAG çıktısında kaynak ile cevabın gerçekten uyuştuğunu ölçün.
- Gecikme, bellek, maliyet, güvenlik ve kalite için ayrı eşikler belirleyin.

## Kaynak ve lisans

Bu çalışma, Maxime Labonne tarafından yayımlanan [mlabonne/llm-course](https://github.com/mlabonne/llm-course) deposunu temel alır. Özgün proje ve bu türetilmiş çalışma [Apache License 2.0](LICENSE) koşulları altında sunulur. Bağlantılı makale, model, veri kümesi ve notebookların kendi lisansları ayrıca geçerlidir.


