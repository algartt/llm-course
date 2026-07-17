# PyTorch ve TensorFlow/Keras Karşılaştırma Raporu

## Deney ortamı

| Alan | Değer |
|---|---|
| Tarih |  |
| Python |  |
| İşletim sistemi |  |
| CPU/GPU |  |
| PyTorch |  |
| TensorFlow |  |
| Keras |  |
| Seed | 42 |

## Ortak ayarlar

| Ayar | Değer |
|---|---:|
| Veri örneği | 48 |
| Test oranı | 0.25 |
| Epoch | 60 |
| Batch | 8 |
| Learning rate | 0.02 |
| Gizli katman | 32 |

## Sonuçlar

| Metrik | PyTorch | TensorFlow/Keras |
|---|---:|---:|
| Accuracy |  |  |
| Precision |  |  |
| Recall |  |  |
| F1 |  |  |
| Eğitim süresi |  |  |
| Model dosyası boyutu |  |  |

## Hata analizi

| Metin | Gerçek | PyTorch | TensorFlow/Keras | Gözlem |
|---|---:|---:|---:|---|
|  |  |  |  |  |

## Yorum

- Hangi model daha kararlı sonuç verdi?
- Eğitim süresi farkının olası nedeni nedir?
- Sonuçlar aynı seed ile tekrarlandığında ne kadar değişti?
- Veri büyütüldüğünde hangi darboğazlar bekleniyor?
- Bu küçük veri kümesi gerçek dünya sonucu hakkında neden yeterli değil?

## Sonraki deney

- Daha büyük ve lisansı açık bir Türkçe veri kümesi kullanın.
- TF-IDF veya subword tokenizer ekleyin.
- Macro-F1 ve sınıf bazlı hata analizini genişletin.
- Aynı modeli 5 farklı seed ile çalıştırıp ortalama ve standart sapma raporlayın.


