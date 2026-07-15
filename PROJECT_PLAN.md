# MS Stock Price Prediction - Proje Plani

## 1. Projenin amaci

Microsoft (`MSFT`) hissesinin gecmis gunluk fiyat verilerini kullanarak bir sonraki islem gununun kapanis fiyatini tahmin eden iki PyTorch zaman serisi modeli gelistirmek ve LSTM ile GRU performansini adil kosullarda karsilastirmak.

Bu proje bir yatirim tavsiyesi veya otomatik al-sat sistemi degildir. Ana hedef; veri hazirlama, zaman serisi dogrulamasi, PyTorch egitim dongusu ve model degerlendirme surecini ogrenmektir.

## 2. Baslangic kapsami

- Hisse: Microsoft (`MSFT`)
- Problem: Tek degiskenli zaman serisi regresyonu
- Girdi: Onceki 20 islem gununun kapanis fiyatlari
- Hedef: Bir sonraki islem gununun kapanis fiyati
- Modeller: Naive baseline, LSTM ve GRU
- Ana metrik: Test RMSE (orijinal fiyat biriminde)
- Yardimci metrikler: MAE, MSE, egitim suresi ve parametre sayisi
- Veri bolme: Zamana gore train/validation/test; rastgele karistirma yok

Bu kapsam ilk calisan surum icindir. Hacim, teknik gostergeler, haberler veya Transformer modelleri daha sonraki asamalara birakilir.

## 3. Su anda elimizde olanlar

### Hazir

- Yerel Git deposu ve `main` dali
- GitHub `origin`: `https://github.com/bsdburak/MS-Stock-Price-Prediction.git`
- Windows uzerinde Python 3.14.4
- `uv` 0.11.7 paket/proje yoneticisi
- Global Python kurulumunda Pandas, NumPy, scikit-learn, Matplotlib ve Jupyter
- Proje hedeflerini ve 30 is gunluk ogrenme akisini aciklayan kaynak PDF

### Eksik

- Projeye ozel sanal ortam ve kilitli bagimliliklar
- PyTorch
- Hisse verisini indirme yontemi/paketi
- Veri dosyalari
- Notebook, kaynak kod, testler ve sonuc grafikleri
- README ve calistirma talimatlari
- Ilk Git commit'i

Not: `python` ve `jupyter` komutlari PATH uzerinden gorunmuyor. Bu nedenle proje komutlari `uv run ...` ile calistirilacak; global kuruluma bagimli olunmayacak.

## 4. Teknik yaklasim

### Veri

1. Tek bir tekrarlanabilir kaynaktan MSFT gunluk OHLCV verisi al.
2. Tarihe gore sirala, yinelenen satirlari ve eksik degerleri kontrol et.
3. Ilk surumde kapanis fiyatini kullan.
4. Veriyi kronolojik olarak train/validation/test bolumlerine ayir.
5. Olcekleyiciyi yalnizca train verisine fit et; validation ve test verisine ayni donusumu uygula.
6. 20 gunluk kayan pencerelerden girdileri ve ertesi gun hedeflerini olustur.

### Modeller

1. Naive baseline: Ertesi gun tahmini olarak son gozlenen fiyati kullan.
2. LSTM: Ayni girdi penceresi ve ayni veri bolumleriyle egit.
3. GRU: LSTM ile mumkun oldugunca ayni hiperparametreleri kullan.
4. Her model icin validation kaybini izle ve en iyi agirliklari sakla.

### Degerlendirme

- Tum metrikleri olcek tersine cevrildikten sonra gercek fiyat biriminde hesapla.
- Test setine yalnizca model ve hiperparametre secimi bittikten sonra bak.
- LSTM ve GRU'yu baseline'a, birbirlerine, egitim suresine ve parametre sayisina gore karsilastir.
- Gercek ve tahmin edilen fiyatlari ayni tarih ekseninde ciz.
- Modelin trendi takip ediyor gorunmesinin tek basina basari olmadigini; baseline'i gecmesi gerektigini belirt.

## 5. Calisma plani - 6 hafta / 30 is gunu

PDF basliginda bir ay yazsa da ayrintili program 30 is gunu, yani yaklasik 6 hafta suruyor. Asagidaki plan belgeyle uyumlu olan gercekci takvimdir.

### Hafta 1 - Python ve ML temelleri

- Proje ortamini kur, Jupyter ve PyTorch kurulumunu dogrula.
- Supervised/unsupervised learning, regresyon/siniflandirma kavramlarini ogren.
- Python ve Pandas ile temel veri islemleri yap.
- Cikti: Ortam dogrulama notebook'u ve kisa Pandas alistirmasi.

### Hafta 2 - Regresyon ve veri analizi

- Train/test ayrimi, MSE, RMSE ve MAE kavramlarini uygula.
- Basit bir scikit-learn regresyon alistirmasi tamamla.
- Zaman serilerinde rastgele bolmenin neden veri sizintisi yaratabilecegini ogren.
- Cikti: Basit regresyon notebook'u ve metrik aciklamalari.

### Hafta 3 - PyTorch ve zaman serileri

- Tensor, Dataset/DataLoader, `nn.Module`, loss, optimizer ve backpropagation calis.
- Basit bir PyTorch egitim dongusu kur.
- RNN, LSTM, GRU, hidden state ve kayan pencere mantigini ogren.
- MSFT veri kaynagini ve sabit veri tarih araligini kararlastir.
- Cikti: PyTorch hizli baslangic notebook'u ve kesinlesmis veri sozlesmesi.

### Hafta 4 - Veri hatti ve model uygulamasi

- MSFT verisini indir, dogrula ve kesifsel grafiklerini uret.
- Kronolojik bolme, train-only scaling ve kayan pencere kodunu yaz.
- Naive baseline'i olustur.
- LSTM ve GRU siniflarini, ortak egitim dongusunu uygula.
- Cikti: Uctan uca calisan ilk pipeline ve ilk egitim sonucu.

### Hafta 5 - Egitim, ayar ve karsilastirma

- LSTM ve GRU'yu ayni veri ve benzer hiperparametrelerle egit.
- Validation setiyle lookback, hidden size, layer sayisi ve learning rate icin sinirli deneyler yap.
- Test RMSE/MAE, egitim suresi ve parametre sayisini raporla.
- Gercek/tahmin grafiklerini olustur ve baseline karsilastirmasini yap.
- Cikti: Sonuc tablosu, grafikler ve secilmis model agirliklari.

### Hafta 6 - Temizlik ve dokumantasyon

- Notebook'u bastan sona temiz ortamda calistir.
- Tekrarlanan kodu `src/` modullerine tasi ve temel testleri ekle.
- README'ye problem, veri, kurulum, calistirma, sonuclar ve sinirlamalari yaz.
- Sonuclari yeniden uretilebilir hale getir ve GitHub'a gonder.
- Cikti: Sunulabilir GitHub deposu ve tamamlanmis LSTM/GRU karsilastirmasi.

## 6. Onerilen depo yapisi

```text
MS-Stock-Price-Prediction/
|-- README.md
|-- PROJECT_PLAN.md
|-- pyproject.toml
|-- uv.lock
|-- .gitignore
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   `-- 02_model_comparison.ipynb
|-- src/
|   |-- data.py
|   |-- models.py
|   |-- train.py
|   `-- evaluate.py
|-- tests/
`-- results/
    |-- figures/
    `-- metrics/
```

Buyuk veya otomatik indirilebilen veri dosyalari Git'e eklenmeyecek. Kaynak, tarih araligi ve yeniden indirme adimi README'de belirtilecek.

## 7. Tamamlanma kriterleri

- Temiz bir ortamda tek komutla bagimliliklar kurulabiliyor.
- Veri indirme ve hazirlama adimlari yeniden calistirilabiliyor.
- Train/validation/test ayrimi kronolojik ve veri sizintisiz.
- Naive baseline, LSTM ve GRU ayni test tarihleri uzerinde degerlendiriliyor.
- RMSE/MAE, egitim suresi ve parametre sayisi tek tabloda bulunuyor.
- Gercek ve tahmin grafiklerinin tarih ekseni dogru.
- Notebook bastan sona hatasiz calisiyor.
- README proje amacini, kurulumu, sonuclari ve sinirlamalari acikliyor.
- Kod ve dokumantasyon GitHub deposunda yer aliyor.

## 8. Siradaki adim

Ilk teknik adim, projeye ozel `uv` ortamini ve depo iskeletini kurmaktir. Ardindan veri kaynagi kesinlestirilip MSFT verisiyle kesifsel analiz notebook'una gecilecektir.
