"""
ML Tahmin Modelleri
===================
Prophet, Random Forest ve Linear Regression ile
kısa/orta vadeli fon fiyat tahmini.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")


class FonTahminModeli:
    """
    Birden fazla ML modeli ile fon fiyat tahmini yapar.

    Kullanım:
        model = FonTahminModeli(fon_df)
        tahmin_df, metrikler = model.tahmin_et("Prophet", 30)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("Tarih").reset_index(drop=True)
        self.df["Tarih"] = pd.to_datetime(self.df["Tarih"])
        self.fiyatlar = self.df["BirimPayDegeri"].values
        self.tarihler = self.df["Tarih"].values

    # ──────────────────────────────────────────────────────────────
    def tahmin_et(self, model_adi: str, gun_sayisi: int):
        """
        Belirtilen model ile tahmin çalıştırır.

        Returns
        -------
        tahmin_df : pd.DataFrame — ds, yhat, yhat_lower, yhat_upper
        metrikler : dict — mape, rmse gibi başarı metrikleri
        """
        if "Prophet" in model_adi:
            return self._prophet_tahmin(gun_sayisi)
        elif "Random Forest" in model_adi:
            return self._rf_tahmin(gun_sayisi)
        elif "Linear" in model_adi:
            return self._linreg_tahmin(gun_sayisi)
        elif "Ensemble" in model_adi:
            return self._ensemble_tahmin(gun_sayisi)
        else:
            return self._prophet_tahmin(gun_sayisi)

    # ──────────────────────────────────────────────────────────────
    # PROPHET MODELİ
    # ──────────────────────────────────────────────────────────────
    def _prophet_tahmin(self, gun_sayisi: int):
        """
        Prophet kütüphanesi varsa kullanır;
        yoksa kendi elle yazılmış decomposition modelimizi çalıştırır.
        """
        try:
            from prophet import Prophet
            return self._prophet_gercek(gun_sayisi)
        except ImportError:
            return self._prophet_benzeri(gun_sayisi)

    def _prophet_gercek(self, gun_sayisi: int):
        """Gerçek Prophet modeli (kütüphane kuruluysa)."""
        from prophet import Prophet

        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(self.tarihler),
            "y":  self.fiyatlar
        })

        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        model.fit(prophet_df)

        gelecek = model.make_future_dataframe(periods=gun_sayisi)
        tahmin  = model.predict(gelecek)

        # Sadece gelecek günleri döndür
        tahmin_gelecek = tahmin[tahmin["ds"] > prophet_df["ds"].max()][
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ]

        # Geriye dönük MAPE hesapla
        gecmis = tahmin[tahmin["ds"] <= prophet_df["ds"].max()]
        mape = mean_absolute_percentage_error(prophet_df["y"], gecmis["yhat"].values[-len(prophet_df):])
        metrikler = {"mape": f"{mape*100:.2f}%", "model": "Prophet"}

        return tahmin_gelecek.reset_index(drop=True), metrikler

    def _prophet_benzeri(self, gun_sayisi: int):
        """
        Prophet kütüphanesi yoksa çalışan manuel implementasyon.
        Trend + mevsimsellik + gürültü ayrıştırması.
        """
        n = len(self.fiyatlar)
        t = np.arange(n)

        # ── Trend: piecewise linear (değişim noktaları)
        n_cp = max(3, n // 50)
        cp_idx = np.linspace(0, n - 1, n_cp + 2, dtype=int)[1:-1]

        # Ridge regresyon ile trend öğren
        X_trend = self._changepoint_matrisi(t, cp_idx)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_trend)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, self.fiyatlar)
        trend = ridge.predict(X_scaled)

        # ── Mevsimsellik (yıllık + haftalık)
        tarihler_dt = pd.to_datetime(self.tarihler)
        yil_periyot = 365.25
        hafta_periyot = 7.0

        mevsim = np.zeros(n)
        for periyot, harmonikler in [(yil_periyot, 10), (hafta_periyot, 3)]:
            for k in range(1, harmonikler + 1):
                mevsim += (
                    np.sin(2 * np.pi * k * t / periyot) +
                    np.cos(2 * np.pi * k * t / periyot)
                ) * 0.001

        # ── Tahmin
        son_cp_idx = cp_idx[-1]
        t_gelecek = np.arange(n, n + gun_sayisi)

        # Trend ekstrapolasyonu
        son_egim = (trend[-1] - trend[max(0, -30)]) / min(30, n)
        trend_gelecek = trend[-1] + son_egim * np.arange(1, gun_sayisi + 1)

        # Mevsim ekstrapolasyonu
        mevsim_gelecek = np.zeros(gun_sayisi)
        for periyot, harmonikler in [(yil_periyot, 10), (hafta_periyot, 3)]:
            for k in range(1, harmonikler + 1):
                mevsim_gelecek += (
                    np.sin(2 * np.pi * k * t_gelecek / periyot) +
                    np.cos(2 * np.pi * k * t_gelecek / periyot)
                ) * 0.001

        yhat = np.maximum(trend_gelecek + mevsim_gelecek, 0.0001)

        # Güven aralığı — sigma sınırlı (uzun vadede patlamaması için)
        getiriler = np.diff(np.log(self.fiyatlar))
        sigma = getiriler.std() * np.sqrt(np.arange(1, gun_sayisi + 1))
        sigma = np.clip(sigma, 0, 0.5)
        yhat_lower = np.maximum(yhat * np.exp(-1.96 * sigma), yhat * 0.5)
        yhat_upper = yhat * np.exp(+1.96 * sigma)

        # MAPE tahmini (in-sample)
        in_sample_pred = trend + mevsim
        mape_val = np.mean(np.abs((self.fiyatlar - in_sample_pred) / self.fiyatlar)) * 100

        son_tarih = pd.to_datetime(self.tarihler[-1])
        ds_gelecek = pd.date_range(
            start=son_tarih + timedelta(days=1),
            periods=gun_sayisi,
            freq="B"
        )

        tahmin_df = pd.DataFrame({
            "ds":         ds_gelecek,
            "yhat":       np.maximum(yhat, 0.0001),
            "yhat_lower": np.maximum(yhat_lower, 0.0001),
            "yhat_upper": np.maximum(yhat_upper, 0.0001),
        })

        metrikler = {"mape": f"{mape_val:.2f}%", "model": "Prophet (manuel)"}
        return tahmin_df, metrikler

    def _changepoint_matrisi(self, t, cp_idx):
        """Piecewise linear değişim noktaları için özellik matrisi."""
        X = np.column_stack([t] + [np.maximum(0, t - cp) for cp in cp_idx])
        return X

    # ──────────────────────────────────────────────────────────────
    # RANDOM FOREST MODELİ
    # ──────────────────────────────────────────────────────────────
    def _rf_tahmin(self, gun_sayisi: int):
        """Zaman serisi özelliklerini kullanarak RF tahmin yapar."""
        df = self._ozellik_olustur()
        hedef_kolon = "BirimPayDegeri"

        # Eğitim/test bölümü
        test_boyutu = min(30, len(df) // 5)
        egitim = df.iloc[:-test_boyutu]
        test   = df.iloc[-test_boyutu:]

        ozellikler = [c for c in df.columns if c != hedef_kolon]
        X_egitim = egitim[ozellikler]
        y_egitim = egitim[hedef_kolon]
        X_test   = test[ozellikler]
        y_test   = test[hedef_kolon]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_egitim, y_egitim)

        # MAPE
        y_pred_test = model.predict(X_test)
        mape_val = mean_absolute_percentage_error(y_test, y_pred_test) * 100

        # Gelecek tahmin — tek adım öngörü, ileri sarma
        son_fiyat = self.fiyatlar[-1]
        son_tarih = pd.to_datetime(self.tarihler[-1])
        tahminler = []
        gecmis    = list(self.fiyatlar[-60:])  # son 60 gün bellekte tut

        for i in range(gun_sayisi):
            gecmis_arr = np.array(gecmis)
            X_gelecek = self._tek_satir_ozellik(gecmis_arr, i)
            tahmin = float(model.predict([X_gelecek])[0])
            tahminler.append(max(tahmin, 0.0001))
            gecmis.append(tahmin)

        ds_gelecek = pd.date_range(start=son_tarih + timedelta(days=1), periods=gun_sayisi, freq="B")

        # Güven aralığı
        sigma = np.std(np.diff(np.log(self.fiyatlar))) * np.sqrt(np.arange(1, gun_sayisi + 1))
        yhat  = np.array(tahminler)
        tahmin_df = pd.DataFrame({
            "ds":         ds_gelecek,
            "yhat":       yhat,
            "yhat_lower": yhat * np.exp(-1.96 * sigma),
            "yhat_upper": yhat * np.exp(+1.96 * sigma),
        })

        metrikler = {"mape": f"{mape_val:.2f}%", "model": "Random Forest"}
        return tahmin_df, metrikler

    # ──────────────────────────────────────────────────────────────
    # LINEAR REGRESSION MODELİ
    # ──────────────────────────────────────────────────────────────
    def _linreg_tahmin(self, gun_sayisi: int):
        """Zaman serisi özellikleriyle Ridge Regression tahmini."""
        df = self._ozellik_olustur()
        hedef_kolon = "BirimPayDegeri"
        ozellikler  = [c for c in df.columns if c != hedef_kolon]

        X = df[ozellikler].values
        y = df[hedef_kolon].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=10.0)
        model.fit(X_scaled, y)

        y_pred = model.predict(X_scaled)
        mape_val = mean_absolute_percentage_error(y, y_pred) * 100

        # Gelecek öngörüsü
        son_tarih = pd.to_datetime(self.tarihler[-1])
        gecmis = list(self.fiyatlar[-60:])
        tahminler = []

        for i in range(gun_sayisi):
            gecmis_arr = np.array(gecmis)
            X_tek = self._tek_satir_ozellik(gecmis_arr, i)
            X_tek_scaled = scaler.transform([X_tek])
            tahmin = float(model.predict(X_tek_scaled)[0])
            tahminler.append(max(tahmin, 0.0001))
            gecmis.append(tahmin)

        ds_gelecek = pd.date_range(start=son_tarih + timedelta(days=1), periods=gun_sayisi, freq="B")
        sigma = np.std(np.diff(np.log(self.fiyatlar))) * np.sqrt(np.arange(1, gun_sayisi + 1))
        yhat  = np.array(tahminler)

        tahmin_df = pd.DataFrame({
            "ds":         ds_gelecek,
            "yhat":       yhat,
            "yhat_lower": yhat * np.exp(-1.96 * sigma),
            "yhat_upper": yhat * np.exp(+1.96 * sigma),
        })

        metrikler = {"mape": f"{mape_val:.2f}%", "model": "Ridge Regression"}
        return tahmin_df, metrikler

    # ──────────────────────────────────────────────────────────────
    # ENSEMBLE MODELİ
    # ──────────────────────────────────────────────────────────────
    def _ensemble_tahmin(self, gun_sayisi: int):
        """Tüm modellerin ağırlıklı ortalamasını alır."""
        tahminler_dict = {}

        for model_adi, metot in [
            ("Prophet",       self._prophet_benzeri),
            ("RandomForest",  self._rf_tahmin),
            ("LinReg",        self._linreg_tahmin),
        ]:
            try:
                df_t, _ = metot(gun_sayisi)
                tahminler_dict[model_adi] = df_t["yhat"].values
                ds = df_t["ds"]
            except Exception:
                continue

        if not tahminler_dict:
            return self._prophet_benzeri(gun_sayisi)

        yhat = np.mean(list(tahminler_dict.values()), axis=0)
        sigma = np.std(np.diff(np.log(self.fiyatlar))) * np.sqrt(np.arange(1, gun_sayisi + 1))

        tahmin_df = pd.DataFrame({
            "ds":         ds,
            "yhat":       yhat,
            "yhat_lower": yhat * np.exp(-1.96 * sigma),
            "yhat_upper": yhat * np.exp(+1.96 * sigma),
        })

        metrikler = {"mape": "N/A (Ensemble)", "model": "Ensemble"}
        return tahmin_df, metrikler

    # ──────────────────────────────────────────────────────────────
    # YARDIMCI METODLAR
    # ──────────────────────────────────────────────────────────────
    def _ozellik_olustur(self) -> pd.DataFrame:
        """Zaman serisi ML özellikleri üretir."""
        df = pd.DataFrame({
            "BirimPayDegeri": self.fiyatlar,
            "Tarih":          pd.to_datetime(self.tarihler)
        })

        # Lag özellikleri
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f"lag_{lag}"] = df["BirimPayDegeri"].shift(lag)

        # Hareketli ortalamalar
        for pencere in [5, 10, 20, 50]:
            df[f"ma_{pencere}"] = df["BirimPayDegeri"].rolling(pencere).mean()
            df[f"std_{pencere}"] = df["BirimPayDegeri"].rolling(pencere).std()

        # Momentum & RSI
        df["getiri_1"] = df["BirimPayDegeri"].pct_change(1)
        df["getiri_5"] = df["BirimPayDegeri"].pct_change(5)
        df["rsi"]      = self._rsi(df["BirimPayDegeri"])

        # Tarih özellikleri
        df["gun_of_week"]  = df["Tarih"].dt.dayofweek
        df["gun_of_year"]  = df["Tarih"].dt.dayofyear
        df["ay"]           = df["Tarih"].dt.month
        df["ceyrek"]       = df["Tarih"].dt.quarter

        # Mevsimsel Fourier terimleri
        t = np.arange(len(df))
        for k in [1, 2]:
            df[f"sin_yil_{k}"] = np.sin(2 * np.pi * k * t / 252)
            df[f"cos_yil_{k}"] = np.cos(2 * np.pi * k * t / 252)

        df = df.dropna().reset_index(drop=True)
        df = df.drop(columns=["Tarih"])
        return df

    def _tek_satir_ozellik(self, gecmis: np.ndarray, adim: int) -> np.ndarray:
        """Tek bir tahmin adımı için özellik vektörü üretir."""
        n = len(gecmis)
        ozellikler = []

        # Lag'lar
        for lag in [1, 2, 3, 5, 10, 20]:
            ozellikler.append(gecmis[-lag] if n >= lag else gecmis[0])

        # Hareketli ortalama & std
        for pencere in [5, 10, 20, 50]:
            slice_ = gecmis[-pencere:] if n >= pencere else gecmis
            ozellikler.append(np.mean(slice_))
            ozellikler.append(np.std(slice_))

        # Getiri
        ozellikler.append(gecmis[-1] / gecmis[-2] - 1 if n >= 2 else 0)
        ozellikler.append(gecmis[-1] / gecmis[-6] - 1 if n >= 6 else 0)
        ozellikler.append(50.0)  # RSI placeholder

        # Tarih
        t = len(self.fiyatlar) + adim
        gun_of_week = t % 5
        gun_of_year = t % 252
        ozellikler += [gun_of_week, gun_of_year, (t // 21) % 12 + 1, (t // 63) % 4 + 1]

        # Fourier
        for k in [1, 2]:
            ozellikler.append(np.sin(2 * np.pi * k * t / 252))
            ozellikler.append(np.cos(2 * np.pi * k * t / 252))

        return np.array(ozellikler)

    @staticmethod
    def _rsi(seri: pd.Series, pencere: int = 14) -> pd.Series:
        """RSI (Relative Strength Index) hesaplar."""
        delta = seri.diff()
        kazanc = delta.clip(lower=0)
        kayip  = -delta.clip(upper=0)
        ort_k  = kazanc.rolling(pencere).mean()
        ort_ka = kayip.rolling(pencere).mean()
        rs     = ort_k / ort_ka.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
