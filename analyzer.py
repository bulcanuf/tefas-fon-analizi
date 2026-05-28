"""
Fon Analiz Modülü
=================
Getiri, risk, korelasyon ve sıralama hesaplamaları.
"""

import pandas as pd
import numpy as np
from typing import Optional


class FonAnalizci:
    """Seçili fonlar için çeşitli performans ve risk metrikleri hesaplar."""

    YILLIK_GUN = 252  # İş günü sayısı

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values(["FonKodu", "Tarih"]).reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────
    # GETIRI METRİKLERİ
    # ──────────────────────────────────────────────────────────────
    def toplam_getiri(self, fon_kodu: str) -> float:
        """Dönem boyunca toplam getiri (%)."""
        fon = self._fon_df(fon_kodu)
        if len(fon) < 2:
            return 0.0
        return (fon["BirimPayDegeri"].iloc[-1] / fon["BirimPayDegeri"].iloc[0] - 1) * 100

    def yillik_getiri(self, fon_kodu: str) -> float:
        """Yıllık bileşik getiri (CAGR, %)."""
        fon = self._fon_df(fon_kodu)
        if len(fon) < 2:
            return 0.0
        gun = (fon["Tarih"].iloc[-1] - fon["Tarih"].iloc[0]).days
        if gun <= 0:
            return 0.0
        toplam = fon["BirimPayDegeri"].iloc[-1] / fon["BirimPayDegeri"].iloc[0]
        return (toplam ** (365 / gun) - 1) * 100

    # ──────────────────────────────────────────────────────────────
    # RİSK METRİKLERİ
    # ──────────────────────────────────────────────────────────────
    def volatilite(self, fon_kodu: str) -> float:
        """Yıllık volatilite (%)."""
        getiriler = self._gunluk_getiriler(fon_kodu)
        if len(getiriler) < 5:
            return 0.0
        return getiriler.std() * np.sqrt(self.YILLIK_GUN) * 100

    def sharpe(self, fon_kodu: str, risksiz_oran: float = 0.30) -> float:
        """
        Sharpe oranı.
        risksiz_oran: TCMB politika faizi yaklaşık değeri (yıllık)
        """
        yillik = self.yillik_getiri(fon_kodu) / 100
        vol    = self.volatilite(fon_kodu) / 100
        if vol == 0:
            return 0.0
        return (yillik - risksiz_oran) / vol

    def max_drawdown(self, fon_kodu: str) -> float:
        """Maksimum geri çekilme (%)."""
        fon = self._fon_df(fon_kodu)
        fiyat = fon["BirimPayDegeri"].values
        if len(fiyat) < 2:
            return 0.0
        doruk    = np.maximum.accumulate(fiyat)
        drawdown = (fiyat - doruk) / doruk * 100
        return drawdown.min()

    def var_95(self, fon_kodu: str) -> float:
        """Günlük VaR %95 güven düzeyinde (%)."""
        getiriler = self._gunluk_getiriler(fon_kodu) * 100
        if len(getiriler) < 20:
            return 0.0
        return float(np.percentile(getiriler, 5))

    def calmar(self, fon_kodu: str) -> float:
        """Calmar oranı = Yıllık getiri / |Max Drawdown|."""
        yillik = self.yillik_getiri(fon_kodu)
        mdd    = abs(self.max_drawdown(fon_kodu))
        return yillik / mdd if mdd > 0 else 0.0

    # ──────────────────────────────────────────────────────────────
    # TOPLU ANALİZ
    # ──────────────────────────────────────────────────────────────
    def risk_metrikleri_tumu(self, fon_kodlari: list) -> pd.DataFrame:
        """Tüm seçili fonlar için risk metrikleri tablosu döndürür."""
        satirlar = []
        for fon in fon_kodlari:
            fon_df = self._fon_df(fon)
            fon_adi = fon_df["FonAdi"].iloc[0] if "FonAdi" in fon_df.columns and len(fon_df) > 0 else fon
            satirlar.append({
                "FonKodu":          fon,
                "FonAdi":           fon_adi[:40] + "..." if len(str(fon_adi)) > 40 else fon_adi,
                "Getiri (%)":       round(self.toplam_getiri(fon), 2),
                "Yıllık Getiri (%)":round(self.yillik_getiri(fon), 2),
                "Volatilite (%)":   round(self.volatilite(fon), 2),
                "Sharpe":           round(self.sharpe(fon), 3),
                "Max Drawdown (%)": round(self.max_drawdown(fon), 2),
                "VaR 95% (%)":      round(self.var_95(fon), 2),
                "Calmar":           round(self.calmar(fon), 3),
            })
        return pd.DataFrame(satirlar)

    def siralama_tablosu(self, fon_kodlari: list, kriter: str) -> Optional[pd.DataFrame]:
        """Seçilen kritere göre sıralanmış fon tablosu."""
        df = self.risk_metrikleri_tumu(fon_kodlari)
        if df.empty:
            return None

        kriter_eslesme = {
            "Dönem Getirisi (%)":     ("Getiri (%)", False),
            "Sharpe Oranı":           ("Sharpe", False),
            "Volatilite (Düşük→Yüksek)": ("Volatilite (%)", True),
            "Max Drawdown":           ("Max Drawdown (%)", False),
        }

        kolon, artan = kriter_eslesme.get(kriter, ("Getiri (%)", False))
        df = df.sort_values(kolon, ascending=artan).reset_index(drop=True)

        # Kolon yeniden adlandır
        df = df.rename(columns={"Getiri (%)": "Dönem Getirisi (%)"})
        return df

    # ──────────────────────────────────────────────────────────────
    # KORELASYON
    # ──────────────────────────────────────────────────────────────
    def korelasyon_matrisi(self, fon_kodlari: list) -> Optional[pd.DataFrame]:
        """Günlük getiri korelasyon matrisi."""
        if len(fon_kodlari) < 2:
            return None

        getiri_dict = {}
        for fon in fon_kodlari:
            fon_df = self._fon_df(fon).set_index("Tarih")
            getiri_dict[fon] = fon_df["BirimPayDegeri"].pct_change() * 100

        birlesik = pd.DataFrame(getiri_dict).dropna()
        if len(birlesik) < 10:
            return None
        return birlesik.corr().round(3)

    # ──────────────────────────────────────────────────────────────
    # AYLIK GETİRİ MATRİSİ (ISI HARİTASI)
    # ──────────────────────────────────────────────────────────────
    def aylik_getiri_matrisi(self, fon_kodu: str) -> Optional[pd.DataFrame]:
        """Yıl × Ay ısı haritası için getiri matrisi üretir."""
        fon_df = self._fon_df(fon_kodu).copy()
        if len(fon_df) < 20:
            return None

        fon_df["Yil"] = fon_df["Tarih"].dt.year
        fon_df["Ay"]  = fon_df["Tarih"].dt.month

        ay_getiri = (
            fon_df.groupby(["Yil", "Ay"])["BirimPayDegeri"]
            .apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 1 else 0)
            .reset_index(name="Getiri")
        )

        pivot = ay_getiri.pivot(index="Yil", columns="Ay", values="Getiri")
        ay_isimleri = {1:"Oca",2:"Şub",3:"Mar",4:"Nis",5:"May",6:"Haz",
                       7:"Tem",8:"Ağu",9:"Eyl",10:"Eki",11:"Kas",12:"Ara"}
        pivot.columns = [ay_isimleri.get(c, c) for c in pivot.columns]
        return pivot.round(2)

    # ──────────────────────────────────────────────────────────────
    # YARDIMCI
    # ──────────────────────────────────────────────────────────────
    def _fon_df(self, fon_kodu: str) -> pd.DataFrame:
        return self.df[self.df["FonKodu"] == fon_kodu].sort_values("Tarih").reset_index(drop=True)

    def _gunluk_getiriler(self, fon_kodu: str) -> pd.Series:
        fon = self._fon_df(fon_kodu)
        return fon["BirimPayDegeri"].pct_change().dropna()
