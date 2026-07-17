# LLM Mühendisi Rotası

Bu rota, LLM tabanlı bir prototipi güvenli, ölçülebilir ve sürdürülebilir bir uygulamaya dönüştürmeye odaklanır. Model seçimi kadar veri akışı, gecikme, maliyet, gözlemlenebilirlik ve güvenlik de önemlidir.

## Öğrenme hedefleri

- Yerel model ile API tabanlı model arasında bilinçli seçim yapmak.
- Kaynak gösteren temel bir RAG sistemi kurmak.
- Retrieval ve generation aşamalarını ayrı değerlendirmek.
- Modeli API olarak sunup gecikme ve hata metriklerini izlemek.
- Prompt injection ve veri sızıntısına karşı savunma katmanları tasarlamak.

## 1. Model çalıştırma

### API veya yerel model

| Kriter | Model API'si | Yerel/açık model |
|---|---|---|
| Başlangıç süresi | Düşük | Model ve altyapı kurulumu gerekir |
| Ölçeklendirme | Sağlayıcı yönetebilir | Ekip yönetir |
| Gizlilik | Sözleşme ve sağlayıcıya bağlı | Tamamen kendi ortamınızda tutulabilir |
| Özelleştirme | Sağlanan özelliklerle sınırlı | Ağırlık, quantization ve sunucu kontrolü |
| Maliyet | Token/istek tabanlı | Donanım ve operasyon maliyeti |

Kararı gerçek trafik tahmini, veri sınıfı ve gecikme hedefiyle verin.

### Prompt sözleşmesi

Promptlar uygulama kodunun bir parçasıdır. Şunları sürümleyin:

- system talimatı,
- giriş/çıkış şeması,
- few-shot örnekleri,
- hata ve boş veri davranışı,
- model ve sampling ayarları,
- prompt sürümü.

Yapısal çıktı bekleniyorsa JSON Schema ile doğrulama yapın; geçersiz çıktıyı doğrudan veritabanına veya araca göndermeyin.

## 2. Vektör depolama

Belge hazırlama akışı:

```text
Kaynaklar → yükleme → temizleme → parçalama → embedding → indeks
```

Her parçayla birlikte metaveri saklayın:

- kaynak kimliği ve URL,
- belge başlığı,
- bölüm/sayfa,
- oluşturulma ve güncellenme tarihi,
- erişim seviyesi,
- embedding model sürümü.

### Parçalama

Sabit karakter sayısı kolaydır fakat başlık ve paragraf bütünlüğünü bozabilir. İyi başlangıç:

1. Belge yapısına göre bölümleme.
2. Aşırı uzun bölümleri recursive splitter ile küçültme.
3. Küçük örtüşme ekleme.
4. Kaynak metaverisini koruma.

Chunk boyutunu retrieval test kümesiyle seçin.

## 3. Temel RAG

```text
Kullanıcı sorusu
      ↓
Sorgu embeddingi
      ↓
Vektör / hibrit arama
      ↓
Top-k belge parçaları
      ↓
İsteğe bağlı reranker
      ↓
Prompt + kaynak bağlamı
      ↓
LLM cevabı + kaynaklar
```

### Minimum güvenilir RAG özellikleri

- Cevapla birlikte kaynak bağlantıları.
- Yetersiz bağlamda cevap vermeme seçeneği.
- Kullanıcının erişemediği belgenin retrieval sonucundan çıkarılması.
- Kaynak metni ile üretilen iddia uyumu kontrolü.
- Retrieval ve generation loglarının ayrı tutulması.

### Değerlendirme

Retrieval:

- Doğru belge top-k içinde mi?
- Gereksiz parçalar ne kadar fazla?
- Hibrit arama semantik aramayı iyileştiriyor mu?

Generation:

- Cevap bağlama dayanıyor mu?
- Soruyu gerçekten yanıtlıyor mu?
- Kaynak doğru iddiaya bağlanmış mı?
- Bilgi yokken uydurma yapıyor mu?

## 4. Gelişmiş RAG

Baseline yetersiz kaldığında sırayla deneyin:

1. Veri temizliği ve chunking düzeltmesi.
2. Göreve uygun embedding modeli.
3. Anahtar kelime + vektör hibrit arama.
4. Reranker.
5. Sorgu yeniden yazma veya çoklu sorgu.
6. Bağlam sıkıştırma.
7. Sorgu yönlendirme ve farklı indeksler.

Her yeni bileşen gecikme ve hata yüzeyi ekler. Offline testte ölçülmeyen bir bileşeni üretime eklemeyin.

## 5. Ajanlar ve araç kullanımı

Ajan, modelin bir hedef için araç seçip sonuçları değerlendirerek birden fazla adım yürütmesidir.

### Güvenli araç tasarımı

- Her araç için dar ve açık parametre şeması.
- Okuma ve yazma araçlarının ayrılması.
- Hassas işlemler için kullanıcı onayı.
- İzinlerin en az yetki ilkesiyle verilmesi.
- Zaman aşımı, tekrar sınırı ve bütçe.
- Araç çıktısının güvenilmeyen veri sayılması.
- Bütün çağrıların denetlenebilir logları.

Modelin serbest biçimli metnini shell veya SQL olarak doğrudan çalıştırmayın. Parametreleri allowlist ve şema ile doğrulayın.

## 6. Çıkarım optimizasyonu

Ölçülecek temel metrikler:

- İlk token süresi (TTFT).
- Token başına süre.
- İstek başına toplam gecikme.
- Token/saniye throughput.
- Eş zamanlı kullanıcı kapasitesi.
- GPU/CPU ve bellek kullanımı.
- İstek başına maliyet.

Optimizasyon sırası:

1. Gereksiz prompt ve bağlam tokenlarını azaltma.
2. Batch ve eş zamanlılık ayarı.
3. Uygun dtype/quantization.
4. KV cache ve prefix cache.
5. Optimize sunucu veya kernel.
6. Daha küçük model ya da model yönlendirme.

Kaliteyi ölçmeden yalnızca hız optimizasyonu yapmayın.

## 7. Dağıtım mimarisi

```text
İstemci
  ↓
API Gateway / kimlik doğrulama
  ↓
Uygulama servisi
  ├─ Prompt ve politika katmanı
  ├─ Retrieval servisi
  ├─ Model sunucusu / sağlayıcı
  └─ Önbellek
  ↓
Log, metrik, trace ve değerlendirme deposu
```

### API kontrol listesi

- İstek ve cevap şemaları.
- Kimlik doğrulama ve rate limit.
- Timeout, retry ve circuit breaker.
- Streaming iptali.
- Hassas veri maskeleme.
- Health/readiness endpointleri.
- Model/prompt sürüm etiketi.
- Kullanıcı geri bildirim kanalı.

## 8. Gözlemlenebilirlik

Üretimde yalnızca hata oranını değil, model davranışını da izleyin:

- Trafik ve gecikme yüzdelikleri.
- Token ve maliyet dağılımı.
- Boş, reddedilen veya geçersiz cevap oranı.
- Retrieval isabeti ve kaynak kullanımı.
- Prompt/model sürümüne göre kalite.
- Güvenlik filtresi olayları.
- Kullanıcı geri bildirimi.

Promptlarda kişisel veya gizli veri bulunabilir. Loglama öncesi maskeleme ve saklama süresi politikası uygulayın.

## 9. Güvenlik

### Temel tehditler

- Prompt injection.
- Sistem promptu veya gizli veri sızıntısı.
- Yetkisiz belge retrievalı.
- Zararlı araç parametreleri.
- Model/embedding tedarik zinciri riski.
- Aşırı kaynak kullanımı ve maliyet saldırıları.

### Savunma katmanları

1. Kimlik ve belge düzeyinde yetkilendirme.
2. Güvenilmeyen içerik ile talimatların ayrılması.
3. Araç allowlisti ve parametre doğrulama.
4. Çıktı şeması ve içerik kontrolleri.
5. Rate limit, token ve adım bütçesi.
6. Kırmızı takım testleri.
7. Olay kaydı ve müdahale planı.

Tek bir system prompt güvenlik sınırı değildir; güvenlik uygulama ve veri katmanlarında uygulanmalıdır.

## 10. Portföy projesi

### Türkçe kaynaklı belge asistanı

Teslim edilecek parçalar:

- Belge yükleme ve temizleme betiği.
- Chunking ve embedding hattı.
- Vektör veya hibrit arama.
- Kaynak gösteren cevap üretimi.
- En az 30 soruluk değerlendirme kümesi.
- Retrieval ve generation metrikleri.
- Basit web/API arayüzü.
- Güvenlik ve veri gizliliği bölümü.
- Docker veya açık kurulum talimatı.

Örnek sonuç tablosu:

| Sürüm | Recall@5 | Kaynağa bağlılık | p95 gecikme | Not |
|---|---:|---:|---:|---|
| Baseline v1 |  |  |  | Vektör arama |
| v2 |  |  |  | Hibrit arama |
| v3 |  |  |  | Reranker |

## Tamamlama ölçütü

- [ ] Baseline RAG sistemini sabit test kümesinde ölçtüm.
- [ ] Cevaplarda kaynak gösterimini doğruladım.
- [ ] Yetkilendirme ve prompt injection testleri yazdım.
- [ ] p50/p95 gecikme, throughput ve maliyet raporladım.
- [ ] Model, prompt, embedding ve indeks sürümlerini kaydettim.

## İlgili sayfalar

- [LLM Temelleri](llm-temelleri.md)
- [LLM Bilim İnsanı Rotası](llm-bilim-insani.md)
- [Framework karşılaştırması](../guides/pytorch-vs-tensorflow.md)
- [Çalıştırılabilir örnekler](../examples/README.md)


