# Veri Sozlesmesi

## Kaynak ve kullanim

- Kaynak araci: `yfinance`
- Saglayici: Yahoo Finance
- Kullanim amaci: Egitim ve arastirma
- Hisse sembolu: `MSFT`
- Frekans: Gunluk (`1d`)
- Baslangic: `2010-01-01` - dahil
- Bitis: Indirmenin calistirildigi gun - haric
- Son dahil edilebilecek tarih: Calistirma gununden onceki son islem gunu
- Fiyat duzeltmesi: `auto_adjust=True`
- Piyasa oncesi/sonrasi: Dahil degil

`yfinance` Yahoo ile baglantili veya Yahoo tarafindan desteklenen resmi bir kutuphane
degildir. Indirilen veriler kisisel, egitim ve arastirma amaciyla kullanilmalidir.

## Beklenen alanlar

| Alan | Aciklama |
| --- | --- |
| `Date` | Islem tarihi ve CSV indeksi |
| `Open` | Duzeltilmis acilis fiyati |
| `High` | Duzeltilmis gun ici en yuksek fiyat |
| `Low` | Duzeltilmis gun ici en dusuk fiyat |
| `Close` | Duzeltilmis model hedefi/kapanis fiyati |
| `Volume` | Gunluk islem hacmi |

## Kalite kontrolleri

- Veri bos olmamali.
- Tum beklenen alanlar bulunmali.
- Tarihler benzersiz ve kronolojik olmali.
- Eksik deger bulunmamali.
- OHLC fiyatlari pozitif, hacim sifir veya pozitif olmali.
- Her satirda `High >= Low` olmali.

Ham CSV dosyasi yeniden indirilebilir oldugu icin Git tarafindan izlenmez. Ayni ada sahip
`.metadata.json` dosyasi ise indirme zamani, gercek tarih araligi, satir sayisi ve CSV
SHA-256 ozetini kaydeder ve Git tarafindan izlenir. Veri kaynagi, parametreleri ve indirme
kodu da depoda tutulur.
