# Türkçe Duygu Analizi: PyTorch ve TensorFlow/Keras

Bu mini proje aynı veri, train/test bölmesi, kelime sözlüğü, model boyutu ve metriklerle iki frameworkü karşılaştırır. Amaç yüksek doğruluk iddiası değil; adil deney tasarımını ve tekrarlanabilir raporlamayı göstermektir.

## Proje yapısı

```text
framework-comparison/
├── data/tr_sentiment.csv
├── common.py
├── train_pytorch.py
├── train_tensorflow.py
├── test_common.py
├── requirements-pytorch.txt
├── requirements-tensorflow.txt
├── REPORT_TEMPLATE.md
└── outputs/                 # Çalıştırma sırasında oluşur
```

## Ortak deney tasarımı

- 24 olumlu ve 24 olumsuz Türkçe örnek.
- Seed değeri 42.
- Sınıf oranını koruyan %75/%25 train/test ayrımı.
- Sözlük yalnızca eğitim verisinden oluşturulur.
- Normalize kelime sıklığı özellikleri.
- 32 birimli gizli katman ve iki sınıflı çıktı.
- Accuracy, precision, recall, F1 ve confusion matrix sayıları.

> Veri kümesi eğitim amaçlı küçük bir örnektir. Gerçek ürün kalitesi veya framework üstünlüğü hakkında sonuç çıkarmak için kullanılamaz.

## Ortak testleri çalıştırma

Testler yalnızca Python standart kütüphanesini kullanır:

```bash
cd projects/framework-comparison
python -m unittest -v test_common.py
```

## PyTorch

Ayrı bir sanal ortam önerilir:

```bash
python -m venv .venv-pytorch
```

Windows PowerShell:

```powershell
.\.venv-pytorch\Scripts\Activate.ps1
python -m pip install --upgrade -r requirements-pytorch.txt
python train_pytorch.py
```

macOS/Linux:

```bash
source .venv-pytorch/bin/activate
python -m pip install --upgrade -r requirements-pytorch.txt
python train_pytorch.py
```

Çıktılar:

- `outputs/pytorch_metrics.json`
- `outputs/pytorch_sentiment.pt`

## TensorFlow/Keras

```bash
python -m venv .venv-tensorflow
```

Windows PowerShell:

```powershell
.\.venv-tensorflow\Scripts\Activate.ps1
python -m pip install --upgrade -r requirements-tensorflow.txt
python train_tensorflow.py
```

macOS/Linux:

```bash
source .venv-tensorflow/bin/activate
python -m pip install --upgrade -r requirements-tensorflow.txt
python train_tensorflow.py
```

Çıktılar:

- `outputs/tensorflow_metrics.json`
- `outputs/tensorflow_sentiment.keras`
- `outputs/tensorflow_vocabulary.json`

## Aynı ayarlarla çalıştırma

```bash
python train_pytorch.py --epochs 60 --batch-size 8 --learning-rate 0.02 --seed 42
python train_tensorflow.py --epochs 60 --batch-size 8 --learning-rate 0.02 --seed 42
```

Sonuç JSON dosyalarındaki metrikleri [rapor şablonuna](REPORT_TEMPLATE.md) aktarın. Eğitim süresi karşılaştırmasında framework başlangıç maliyetini, cihazı ve paket sürümlerini mutlaka yazın.

## Neden ortak kod kullanılıyor?

Framework karşılaştırmalarında veri bölmesi veya tokenizer değişirse ölçülen fark yalnızca frameworke ait olmaz. `common.py` şu işlemleri iki eğitim betiği için tek yerde yapar:

- CSV doğrulama,
- Türkçe karakterleri koruyan tokenizasyon,
- stratified train/test ayrımı,
- eğitim verisinden sözlük oluşturma,
- vektörleştirme,
- ortak metrik hesabı,
- JSON sonuç kaydı.

## Sınırlamalar

- Veri kümesi çok küçüktür.
- Cümleler kısa ve belirgin duygu kelimeleri içerir.
- Bag-of-words kelime sırasını ve bağlamı kullanmaz.
- Tek train/test bölmesi sonuç varyansını göstermez.
- GPU benchmarkı için model aşırı küçüktür.

## Portföy için geliştirme önerileri

1. Lisansı açık gerçek bir Türkçe veri kümesi ekleyin.
2. Beş farklı seed ile ortalama ve standart sapma raporlayın.
3. TF-IDF baseline ile karşılaştırın.
4. Aynı çok dilli Transformer checkpointini iki frameworkte deneyin.
5. Confusion matrix ve öğrenme eğrisi grafikleri ekleyin.
6. GitHub Actions ile standart kütüphane testlerini otomatik çalıştırın.
7. Sonuçları küçük bir Streamlit veya Gradio arayüzünde gösterin.

## Resmi kaynaklar

- [PyTorch veri yükleme belgeleri](https://docs.pytorch.org/docs/stable/data.html)
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [Keras Sequential API](https://keras.io/api/models/sequential/)
- [Keras Adam API](https://keras.io/api/optimizers/adam/)


