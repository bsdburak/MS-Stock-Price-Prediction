# Nihai Deney Seçimi ve Test Protokolü

Bu belge, validation deneyleri tamamlandıktan sonra nihai test değerlendirmesinde
kullanılacak ayarları dondurur. Bu karar hazırlanırken test dönemine ait tahmin veya
metrik kullanılmamıştır.

## Tahmin görevi

- Hisse: Microsoft (`MSFT`)
- Özellik: düzeltilmiş günlük `Close` fiyatı
- Hedef: önceki işlem günlerinden bir sonraki işlem gününün kapanış fiyatını tahmin etmek
- Veri anlık görüntüsü: `2010-01-04` - `2026-07-14`, 4.156 gözlem
- Bölünme: kronolojik `%70 train / %15 validation / %15 test`
- Validation hedefleri: `2021-07-26` - `2024-01-16`, 623 gün
- Test hedefleri: `2024-01-17` - `2026-07-14`, 624 gün

## Dondurulan ortak ayarlar

| Ayar | Değer |
|---|---:|
| Lookback | 20 işlem günü |
| Girdi / çıktı boyutu | 1 / 1 |
| Gizli birim | 32 |
| Katman | 2 |
| Batch size | 32 |
| Epoch bütçesi | 100 |
| Kayıp fonksiyonu | MSE |
| Optimizer | Adam |
| Öğrenme oranı | 0.001 |
| Rastgelelik tohumu | 42 |
| Cihaz | CPU |

MinMaxScaler `[-1, 1]` aralığında yalnızca train dönemiyle eğitilir. Model ağırlıkları,
100 epoch içinde en düşük validation MSE değerini üreten epoch'tan seçilir. LSTM ve GRU
aynı veri, ön işleme ve eğitim koşullarıyla değerlendirilir.

## Validation kanıtları

| Deney | Validation RMSE | Validation MAE | Seçilen epoch |
|---|---:|---:|---:|
| Naive baseline | 4.9875 USD | 3.7967 USD | - |
| LSTM, 50 epoch bütçesi | 6.8591 USD | 5.4549 USD | 49 |
| LSTM, 100 epoch bütçesi | 5.4265 USD | 4.2340 USD | 96 |
| GRU, 50 epoch bütçesi | 6.6895 USD | 5.2316 USD | 50 |
| GRU, 100 epoch bütçesi | 6.1166 USD | 4.8147 USD | 72 |

### Seçim gerekçeleri

- Epoch bütçesini 50'den 100'e çıkarmak iki modeli de iyileştirdi.
- LSTM, 100 epoch bütçesinde GRU'dan daha düşük validation hatası verdi.
- 10 ve 20 günlük lookback sonuçları neredeyse eşitti; 20 gün 0.0028 USD daha düşük
  RMSE verdi ve proje belgesindeki başlangıç tasarımıyla uyumlu kaldı.
- 16 gizli birim, parametre sayısını azaltmasına rağmen LSTM RMSE değerini 7.1971 USD'ye
  yükseltti. Bu nedenle 32 gizli birim korundu.
- Tek çalıştırmadan ölçülen eğitim süresi model seçimi ölçütü yapılmadı.

## Nihai test protokolü

1. LSTM ve GRU, yukarıdaki dondurulmuş ayarlarla ve seed `42` ile yeniden eğitilir.
2. Her model için en düşük validation kaybındaki ağırlıklar geri yüklenir.
3. Değiştirilmemiş test döneminde LSTM, GRU ve naive baseline için yalnızca bir kez
   MSE, RMSE ve MAE hesaplanır.
4. Model parametre sayıları, gözlenen eğitim süreleri ve test tahmin grafikleri raporlanır.
5. Test sonuçları hiperparametre seçmek veya modeli yeniden ayarlamak için kullanılmaz.
   Beklenenden kötü sonuçlar da değiştirilmeden raporlanır.

Model ağırlıkları yeniden üretilebilir çıktılardır ve `models/` altında yerel olarak
saklanabilir. Bu klasördeki üretilmiş ağırlık dosyaları Git tarafından izlenmez.

## Test kapısı durumu

Bu belgenin oluşturulduğu noktada test dönemine ait model tahmini veya performans metriği
hesaplanmamıştır. Nihai değerlendirme, bu ayarların değiştirilmemesi koşuluyla yapılacaktır.
