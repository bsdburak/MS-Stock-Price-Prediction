# MS Stock Price Prediction

Microsoft (`MSFT`) hissesinin bir sonraki islem gunu kapanis fiyatini tahmin etmek icin
PyTorch LSTM ve GRU modellerini karsilastiran bir zaman serisi ogrenme projesi.

## Proje durumu

Proje baslangic asamasindadir. Ayrintili kapsam ve 30 is gunluk yol haritasi icin
[PROJECT_PLAN.md](PROJECT_PLAN.md) dosyasina bakin.

## Ilk kapsam

- Onceki 20 islem gununden bir sonraki kapanis fiyatini tahmin etme
- Kronolojik train/validation/test ayrimi
- Naive baseline, LSTM ve GRU karsilastirmasi
- RMSE, MAE, egitim suresi ve parametre sayisi raporu
- Gercek ve tahmin edilen fiyat grafigi

## Ortam kurulumu

Gereksinimler: Git ve `uv`.

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__)"
uv run jupyter lab
```

## Testler

```powershell
uv run pytest
```

## Uyari

Bu proje egitim amaclidir; yatirim tavsiyesi veya otomatik al-sat sistemi degildir.
