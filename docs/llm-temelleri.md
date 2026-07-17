# LLM Temelleri

Bu bölüm, büyük dil modellerine geçmeden önce gerekli matematik, Python, derin öğrenme ve doğal dil işleme temellerini uygulamalarla açıklar. Her konuyu bütünüyle bitirmeyi beklemek yerine, proje sırasında ihtiyaç duydukça geri dönmek daha verimlidir.

## Öğrenme hedefleri

Bu bölümü tamamladığınızda:

- tensör şekillerini ve matris çarpımını takip edebilir,
- gradyan inişi ile geri yayılımın görevini açıklayabilir,
- eğitim, doğrulama ve test ayrımını doğru kurabilir,
- metni tokenlara ve sayısal temsillere dönüştürebilir,
- küçük bir sinir ağını eğitip sonuçlarını yorumlayabilirsiniz.

## 1. Makine öğrenmesi için matematik

### Lineer cebir

Dil modellerinin ağırlıkları, aktivasyonları ve token temsilleri tensörlerde tutulur. En önemli kavramlar:

- **Vektör:** Bir tokenın veya örneğin sayısal temsili.
- **Matris:** Birden fazla vektörü veya doğrusal dönüşümü tutar.
- **İç çarpım:** İki temsil arasındaki uyumu ölçer; attention skorlarının temelidir.
- **Matris çarpımı:** Katmanların girdiyi yeni bir uzaya dönüştürmesini sağlar.
- **Norm:** Vektör büyüklüğünü ölçer; normalizasyon ve benzerlik hesaplarında kullanılır.

Transformer attention için temel boyut akışı:

```text
Girdi X:       [batch, token, hidden]
Q, K, V:       [batch, head, token, head_dim]
Q × Kᵀ:        [batch, head, token, token]
Attention × V: [batch, head, token, head_dim]
```

Bir katmanda hata ayıklarken ilk kontrol her zaman tensör şekilleri olmalıdır.

### Türev ve optimizasyon

Model eğitimi, kayıp fonksiyonunu küçülten ağırlıkları arar:

```text
yeni_ağırlık = eski_ağırlık - öğrenme_oranı × gradyan
```

- **Türev:** Küçük bir değişimin sonucu nasıl etkilediğini gösterir.
- **Gradyan:** Çok boyutlu parametre uzayındaki değişim yönüdür.
- **Zincir kuralı:** Katmanlar boyunca türevlerin birleştirilmesini sağlar.
- **Geri yayılım:** Kayıptan başlayarak bütün eğitilebilir ağırlıkların gradyanını hesaplar.
- **AdamW:** Uyarlanabilir öğrenme oranını weight decay ile birleştiren yaygın optimizerdır.

Öğrenme oranı çok yüksekse eğitim kararsızlaşabilir; çok düşükse ilerleme aşırı yavaşlar.

### Olasılık ve istatistik

Dil modeli her adımda olası sonraki tokenlar için bir dağılım üretir.

- **Softmax:** Logitleri toplamı 1 olan olasılıklara dönüştürür.
- **Çapraz entropi:** Doğru tokena verilen olasılığı temel alan kayıp fonksiyonudur.
- **Beklenen değer ve varyans:** Dağılımın merkezi ile yayılımını açıklar.
- **Örnekleme:** Model dağılımından token seçme işlemidir.
- **Güven aralığı:** Bir değerlendirme skorundaki belirsizliği raporlamaya yardım eder.

Tek bir test skoru yerine farklı veri alt grupları ve güven aralıklarıyla raporlama yapın.

## 2. Python ve veri araçları

Öncelikli konular:

1. Değişkenler, listeler, sözlükler ve fonksiyonlar.
2. Sınıflar, modüller ve paket yönetimi.
3. Dosya okuma, JSON/CSV işleme ve hata yönetimi.
4. NumPy dizileri ve vektörleştirilmiş işlemler.
5. pandas ile veri temizleme ve keşif.
6. Matplotlib/Seaborn ile grafikler.
7. scikit-learn ile train/test bölme ve metrikler.

Basit ve tekrarlanabilir bir deney yapısı:

```python
from pathlib import Path
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

Seed belirlemek tek başına tam determinizm sağlamaz; donanım, paralel işlemler ve kullanılan kernel sonuçları etkileyebilir.

## 3. Sinir ağları

Bir sinir ağı genel olarak şu parçaları içerir:

- **Girdi:** Modelin işlediği sayısal özellikler.
- **Ağırlık ve bias:** Eğitim sırasında güncellenen parametreler.
- **Aktivasyon:** Doğrusal olmayan dönüşüm; ReLU, GELU veya SiLU örnek verilebilir.
- **Kayıp:** Tahmin ile hedef arasındaki farkı ölçer.
- **Optimizer:** Gradyanları kullanarak ağırlıkları günceller.

### Eğitim döngüsü

```text
1. Batch al
2. İleri geçiş yap
3. Kaybı hesapla
4. Gradyanları sıfırla
5. Geri yayılım yap
6. Ağırlıkları güncelle
7. Metrikleri kaydet
```

### Aşırı öğrenme

Model eğitim verisini ezberleyip yeni örneklere genelleyemiyorsa overfitting oluşur. Belirtileri:

- Eğitim kaybı düşerken doğrulama kaybının yükselmesi.
- Eğitim skorunun test skorundan çok daha yüksek olması.
- Küçük girdi değişikliklerinde kararsız tahminler.

Çözümler arasında daha fazla/temiz veri, weight decay, dropout, early stopping ve daha küçük model bulunur.

## 4. Doğal dil işleme

### Metin temizleme

Her görev agresif temizlik istemez. Büyük-küçük harf, noktalama, emoji ve yazım hataları görev açısından anlam taşıyabilir. Temizleme kararını veri incelemesine göre verin.

### Tokenizasyon

Tokenizer metni modelin bildiği parçalara ayırıp token kimliklerine dönüştürür. Modern LLM'lerde kelime-altı yöntemler yaygındır.

Dikkat edilmesi gerekenler:

- Türkçe eklemeli bir dil olduğu için tek kelime birden fazla tokena bölünebilir.
- Aynı metin farklı tokenizerlarda farklı uzunluk üretir.
- Token sınırı karakter sınırı değildir.
- Sohbet modellerinde özel chat template kullanılmalıdır.
- Padding ve truncation davranışı açıkça ayarlanmalıdır.

### Embedding

Embedding, token veya metni yoğun bir vektöre dönüştürür. Benzerlik aramasında sık kullanılan cosine similarity:

```text
cosine(a, b) = (a · b) / (||a|| × ||b||)
```

Embedding kalitesini yalnızca birkaç örnekle değil, göreve uygun etiketli sorgu-belge çiftleriyle ölçün.

### Attention

Self-attention, her tokenın dizideki diğer tokenlardan ne kadar bilgi alacağını hesaplar:

```text
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

- `Q` ne arandığını,
- `K` hangi bilginin bulunduğunu,
- `V` taşınacak içeriği temsil eder.

Nedensel dil modellerinde gelecek tokenlara bakmayı engellemek için causal mask uygulanır.

## 5. Değerlendirme temelleri

| Görev | Başlangıç metrikleri |
|---|---|
| İkili sınıflandırma | Accuracy, precision, recall, F1, confusion matrix |
| Çok sınıflı sınıflandırma | Macro-F1, weighted-F1, sınıf bazlı recall |
| Bilgi getirme | Recall@k, Precision@k, MRR, nDCG |
| Metin üretimi | Göreve özel doğruluk, insan değerlendirmesi, güvenlik ve tutarlılık |
| RAG | Context recall/precision, faithfulness, answer relevancy |

Metrik seçimi iş hedefiyle uyumlu olmalıdır. Örneğin riskli içerik tespitinde yanlış negatifler, genel accuracy skorundan daha önemli olabilir.

## 6. Uygulamalar

### Alıştırma 1: Tensör şekilleri

- İki batch, sekiz token, 64 hidden dimension içeren tensör oluşturun.
- Dört attention head için `head_dim` değerini hesaplayın.
- Tensörü `[batch, head, token, head_dim]` şekline dönüştürün.

### Alıştırma 2: Baseline sınıflandırıcı

- Türkçe bir metin veri kümesi seçin.
- TF-IDF + lojistik regresyon baseline oluşturun.
- Macro-F1 ve confusion matrix raporlayın.
- En az 20 hatalı örneği elle sınıflandırın.

### Alıştırma 3: Tokenizer karşılaştırması

- Aynı 100 Türkçe cümleyi iki farklı tokenizerla işleyin.
- Ortalama token sayısını ve en çok parçalanan kelimeleri karşılaştırın.
- Bağlam penceresi maliyetine etkisini açıklayın.

## Tamamlama ölçütü

- [ ] Matris çarpımındaki boyutları açıklayabiliyorum.
- [ ] Gradyan, optimizer ve öğrenme oranının görevini biliyorum.
- [ ] Veri sızıntısına yol açmadan train/test ayrımı yapabiliyorum.
- [ ] Tokenizer, embedding ve attention arasındaki ilişkiyi açıklayabiliyorum.
- [ ] En az bir baseline model eğitip hata analizi yaptım.

## Sonraki adım

- [LLM Bilim İnsanı Rotası](llm-bilim-insani.md)
- [PyTorch ile LLM Uygulamaları](../guides/pytorch-llm.md)
- [TensorFlow ve Keras ile LLM Uygulamaları](../guides/tensorflow-keras-llm.md)


