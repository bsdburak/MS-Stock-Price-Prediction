# MS Stock Price Prediction

Microsoft (`MSFT`) hissesinin bir sonraki işlem günü kapanış fiyatını tahmin etmek için
PyTorch ile geliştirilen LSTM ve GRU modellerini naive baseline ile karşılaştıran zaman
serisi regresyon projesi.

Proje; veri indirme ve doğrulama, keşifsel analiz, veri sızıntısını önleyen ön işleme,
PyTorch model eğitimi, sınırlı hiperparametre deneyleri ve saklı test döneminde nihai
değerlendirme adımlarını içerir.

> Bu proje eğitim ve araştırma amaçlıdır. Yatırım tavsiyesi veya otomatik al-sat sistemi
> değildir.

## Nihai sonuç

Değiştirilmemiş test dönemi `2024-01-17` - `2026-07-14` arasındaki 624 işlem gününü
kapsar. Hiperparametreler test seti açılmadan önce validation sonuçlarıyla dondurulmuştur.

| Model | Test MSE | Test RMSE | Test MAE |
|---|---:|---:|---:|
| Naive baseline | 44.2748 USD² | **6.6539 USD** | **4.6795 USD** |
| LSTM | 436.1931 USD² | 20.8852 USD | 15.2420 USD |
| GRU | 1,376.3479 USD² | 37.0992 USD | 32.0546 USD |

Naive baseline, her gün için bir önceki kapanış fiyatını tahmin olarak kullanır ve iki
sinir ağı modelinden de daha düşük hata üretmiştir. LSTM, GRU'dan daha iyi sonuç verse de
test dönemindeki yüksek fiyat seviyelerini sistematik olarak düşük tahmin etmiştir. Bu
sonuç, günlük fiyat seviyelerinde basit bir baseline'ın güçlü olabileceğini ve düşük
validation hatasının farklı bir piyasa dönemine genellemeyi garanti etmediğini gösterir.

Nihai deney kararı ve test protokolü için
[results/EXPERIMENT_SELECTION.md](results/EXPERIMENT_SELECTION.md) dosyasına bakın.

## Tahmin görevi ve yöntem

| Bileşen | Seçim |
|---|---|
| Hisse | Microsoft (`MSFT`) |
| Özellik | Otomatik düzeltilmiş günlük `Close` fiyatı |
| Hedef | Bir sonraki işlem gününün kapanış fiyatı |
| Lookback | Önceki 20 işlem günü |
| Bölünme | Kronolojik `%70 train / %15 validation / %15 test` |
| Ölçekleme | Train döneminde uydurulan MinMaxScaler `[-1, 1]` |
| Modeller | 2 katmanlı LSTM ve GRU, 32 gizli birim |
| Eğitim | Adam `0.001`, MSE loss, en fazla 100 epoch |
| Model seçimi | En düşük validation MSE değerindeki ağırlıklar |
| Tekrarlanabilirlik | Seed `42`, CPU |

Train, validation ve test hedefleri zaman sırası korunarak ayrılır. Validation ve test
dönemlerinin ilk hedefleri için gereken 20 günlük geçmiş bağlam önceki dönemden alınabilir;
ancak hedef tarihi hangi döneme aitse örnek o dönemde tutulur. Ölçekleyici yalnızca train
verisiyle eğitildiği için gelecekteki fiyat bilgisi ön işlemeye sızmaz.

## Veri

Veri `yfinance` aracılığıyla Yahoo Finance'tan günlük ve `auto_adjust=True` seçeneğiyle
indirilir. Notebook çıktılarında kullanılan sabit veri anlık görüntüsü:

- Tarih aralığı: `2010-01-04` - `2026-07-14`
- Gözlem sayısı: `4,156`
- Eksik veya yinelenen tarih: yok
- Model özelliği: `Close`

Ham CSV yeniden üretilebilir ve zamanla değişebilir olduğu için Git tarafından izlenmez.
İzlenen metadata dosyası tarih aralığını, satır sayısını ve CSV SHA-256 özetini kaydeder.
Bugün yeniden indirilen veri daha yeni işlem günleri içerebilir; bu durumda bölünmeler ve
metrikler notebook'taki kayıtlı sonuçlardan farklı olabilir. Ayrıntılı veri sözleşmesi için
[data/README.md](data/README.md) dosyasına bakın.

## Kurulum

Gereksinimler:

- Git
- [uv](https://docs.astral.sh/uv/)
- Python `3.14` (uv tarafından kurulabilir)

```powershell
git clone https://github.com/bsdburak/MS-Stock-Price-Prediction.git
cd MS-Stock-Price-Prediction
uv sync
uv run python -c "import torch; print(torch.__version__)"
```

## Veriyi indirme

```powershell
uv run download-msft-data
```

Komut, `2010-01-01` tarihinden çalıştırıldığı günün öncesindeki son tamamlanmış işlem
gününe kadar olan veriyi `data/raw/msft_daily_latest.csv` dosyasına yazar. Aynı klasördeki
`msft_daily_latest.metadata.json` dosyası indirme bilgilerini saklar.

## Notebook çalışma sırası

JupyterLab'i başlatmak için:

```powershell
uv run jupyter lab
```

Notebook'lar aşağıdaki sırayla okunabilir ve çalıştırılabilir:

1. [`01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) — veri kalitesi,
   betimsel istatistikler ve fiyat/hacim grafikleri.
2. [`02_data_preparation.ipynb`](notebooks/02_data_preparation.ipynb) — kronolojik
   bölünme, train-only ölçekleme, 20 günlük pencereler, tensor ve DataLoader üretimi.
3. [`03_model_training.ipynb`](notebooks/03_model_training.ipynb) — LSTM eğitimi ve
   validation tabanlı ağırlık seçimi.
4. [`04_gru_training.ipynb`](notebooks/04_gru_training.ipynb) — aynı koşullarda GRU
   eğitimi.
5. [`05_validation_comparison.ipynb`](notebooks/05_validation_comparison.ipynb) — ilk
   LSTM, GRU ve naive validation karşılaştırması.
6. [`06_epoch_budget_experiment.ipynb`](notebooks/06_epoch_budget_experiment.ipynb) —
   50 ve 100 epoch bütçelerinin kontrollü karşılaştırması.
7. [`07_lookback_experiment.ipynb`](notebooks/07_lookback_experiment.ipynb) — 10 ve 20
   günlük lookback deneyi.
8. [`08_hidden_size_experiment.ipynb`](notebooks/08_hidden_size_experiment.ipynb) — 16
   ve 32 gizli birim deneyi.
9. [`09_final_test_evaluation.ipynb`](notebooks/09_final_test_evaluation.ipynb) —
   dondurulan modellerin validation kontrolü ve nihai test değerlendirmesi.

Notebook'lar çalıştırılmış çıktılarıyla birlikte depoda tutulur. Son notebook yeniden
çalıştırıldığında seçilen ağırlıkları `models/lstm_selected.pt` ve
`models/gru_selected.pt` altında üretir; bu dosyalar Git tarafından izlenmez.

## Depo yapısı

```text
.
├── data/                   # Veri sözleşmesi, metadata ve yerel ham/işlenmiş veri
├── models/                 # Yerel, yeniden üretilebilir model ağırlıkları
├── notebooks/              # EDA, hazırlık, eğitim, tuning ve değerlendirme
├── results/                # Dondurulan deney seçimi ve rapor belgeleri
├── src/ms_stock_prediction/
│   ├── data.py             # Veri indirme, doğrulama ve metadata üretimi
│   ├── models.py           # LSTM ve GRU sınıfları
│   ├── preprocessing.py    # Tekrar kullanılabilir zaman serisi veri hattı
│   └── training.py         # Eğitim ve değerlendirme yardımcıları
├── tests/                  # Veri, model, ön işleme ve eğitim testleri
├── pyproject.toml          # Bağımlılıklar ve araç ayarları
└── uv.lock                 # Kilitlenmiş ortam
```

## Test ve kod kalitesi

```powershell
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

Mevcut test paketi; veri doğrulama, veri indirme yardımcıları, kronolojik ön işleme,
kayan pencereler, DataLoader davranışı, model çıktı şekilleri ve eğitim/değerlendirme
fonksiyonlarını kapsar.

## Sınırlamalar

- Çalışma yalnızca tek hisse ve tek özellik (`Close`) kullanır.
- Fiyat seviyesi durağan değildir; train ve test dönemleri arasında belirgin dağılım
  değişimi vardır.
- Modeller haber, bilanço, makroekonomik veri veya işlem hacmi gibi ek açıklayıcı
  özellikleri kullanmaz.
- Sonuçlar tek kronolojik bölünme ve tek seed'e dayanır; gelecekte aynı performansı
  garanti etmez.
- Düşük fiyat tahmin hatası, kârlı veya uygulanabilir bir yatırım stratejisi anlamına
  gelmez.

Bu sınırlamalar nedeniyle proje, tahmin başarısından çok doğru zaman serisi deney tasarımı,
veri sızıntısını önleme, güçlü baseline kullanma ve başarısız sonuçları dürüstçe raporlama
üzerine bir öğrenme çalışması olarak değerlendirilmelidir.
