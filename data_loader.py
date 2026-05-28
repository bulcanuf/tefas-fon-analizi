"""
TEFAS Veri Yükleme Modülü
=========================
- Otomatik TEFAS API çekimi
- Manuel CSV yükleme
- Demo veri üretici
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import io
import time
import json


class TEFASDataLoader:
    """TEFAS'tan fon verileri çeken ve düzenleyen sınıf."""

    TEFAS_BASE = "https://www.tefas.gov.tr"
    TEFAS_API  = "https://www.tefas.gov.tr/api/DB/BindHistoryInfo"

    # Altın & Emtia fon kategorisi kodları
    KATEGORI_KODLARI = {
        "Altın Fonu":                "YAL",
        "Emtia Fonu":                "YEM",
        "Kıymetli Madenler":         "YMD",
        "Altın Borsa Yatırım Fonu":  "YBE",
    }

    DEMO_FONLAR = {
        "ATA":  ("Ata Portföy Altın Fonu",          "Altın Fonu"),
        "GAF":  ("Garanti Altın Fonu",               "Altın Fonu"),
        "ING":  ("İng Portföy Kıymetli Madenler",   "Kıymetli Madenler"),
        "ZLT":  ("Ziraat Altın Fonu",                "Altın Fonu"),
        "YDA":  ("Yapı Kredi Altın Fonu",            "Altın Fonu"),
        "EMT1": ("TEB Emtia Fonu",                   "Emtia Fonu"),
    }

    # ──────────────────────────────────────────────────────────────
    def tefas_api_cek(self, fon_turleri: list, tarih_aralik) -> pd.DataFrame:
        """
        TEFAS web sitesinden belirlenen fon türleri için tarihsel veri çeker.

        Parameters
        ----------
        fon_turleri : list  — Kullanıcının seçtiği fon türleri
        tarih_aralik : tuple — (başlangıç, bitiş) date nesneleri

        Returns
        -------
        pd.DataFrame — Standart formatta fon verisi
        """
        bas, bit = tarih_aralik[0], tarih_aralik[1]
        bas_str  = bas.strftime("%d.%m.%Y")
        bit_str  = bit.strftime("%d.%m.%Y")

        # Seçilen kategorilerin kodlarını al
        kat_kodlari = [
            self.KATEGORI_KODLARI[t]
            for t in fon_turleri
            if t in self.KATEGORI_KODLARI
        ] or list(self.KATEGORI_KODLARI.values())

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": self.TEFAS_BASE + "/TarihselVeriler.aspx"
        }

        tum_veriler = []

        for kat_kod in kat_kodlari:
            try:
                payload = {
                    "fontip": kat_kod,
                    "bastarih": bas_str,
                    "bittarih": bit_str,
                    "fonkodu": "",
                }
                resp = requests.post(self.TEFAS_API, data=payload, headers=headers, timeout=30)

                if resp.status_code != 200:
                    continue

                data = resp.json()
                if not data or "data" not in data:
                    continue

                for kayit in data["data"]:
                    tum_veriler.append({
                        "Tarih":           pd.to_datetime(kayit.get("TARIH", ""), dayfirst=True, errors="coerce"),
                        "FonKodu":         kayit.get("FONKODU", ""),
                        "FonAdi":          kayit.get("FONUNVANI", ""),
                        "BirimPayDegeri":  float(kayit.get("BIRIMPAYDEGERI", 0) or 0),
                        "ToplamDeger":     float(kayit.get("PORTFOYBUYUKLUGU", 0) or 0),
                        "YatirimciSayisi": int(kayit.get("YATIRIMCISAYISI", 0) or 0),
                        "FonTuru":         kat_kod,
                    })

                time.sleep(0.5)  # sunucuya saygı

            except Exception:
                continue  # bir kategori başarısız olursa devam et

        if not tum_veriler:
            # API başarısız → demo veriye geri dön
            return self.demo_veri_olustur(fon_turleri)

        df = pd.DataFrame(tum_veriler)
        df = df.dropna(subset=["Tarih", "FonKodu"])
        df = df[df["BirimPayDegeri"] > 0]
        df = df.sort_values(["FonKodu", "Tarih"]).reset_index(drop=True)
        return df

    # ──────────────────────────────────────────────────────────────
    def csv_yukle(self, dosyalar) -> pd.DataFrame:
        """
        Kullanıcının TEFAS'tan indirdiği CSV/Excel dosyalarını yükler.

        Desteklenen formatlar:
        1. TEFAS standart CSV çıktısı
        2. Fon bazlı birden fazla dosya (otomatik birleştirilir)
        """
        parcalar = []

        for dosya in dosyalar:
            try:
                if dosya.name.endswith(".xlsx"):
                    icerik = pd.read_excel(dosya, engine="openpyxl")
                else:
                    # Encoding dene: UTF-8 → Latin-1
                    try:
                        icerik = pd.read_csv(dosya, encoding="utf-8", sep=None, engine="python")
                    except UnicodeDecodeError:
                        dosya.seek(0)
                        icerik = pd.read_csv(dosya, encoding="latin-1", sep=None, engine="python")

                icerik = self._standartlastir(icerik, dosya.name)
                parcalar.append(icerik)

            except Exception as e:
                print(f"Dosya yükleme hatası ({dosya.name}): {e}")
                continue

        if not parcalar:
            raise ValueError("Hiçbir dosya okunamadı. TEFAS CSV formatını kontrol edin.")

        df = pd.concat(parcalar, ignore_index=True)
        df = df.drop_duplicates(subset=["FonKodu", "Tarih"])
        df = df.sort_values(["FonKodu", "Tarih"]).reset_index(drop=True)
        return df

    def _standartlastir(self, df: pd.DataFrame, dosya_adi: str) -> pd.DataFrame:
        """TEFAS CSV formatını standart formata dönüştürür."""
        # Kolon isimlerini normalize et
        df.columns = [str(c).strip() for c in df.columns]

        eslesme = {
            # Tarih
            "Tarih": "Tarih",
            "tarih": "Tarih",
            "DATE": "Tarih",
            # Fon Kodu
            "Fon Kodu": "FonKodu",
            "FON KODU": "FonKodu",
            "FONKODU": "FonKodu",
            # Fon Adı
            "Fon Adı": "FonAdi",
            "FON ADI": "FonAdi",
            "FONUNVANI": "FonAdi",
            # Birim Pay Değeri
            "Birim Pay Değeri": "BirimPayDegeri",
            "BPD": "BirimPayDegeri",
            "BIRIMPAYDEGERI": "BirimPayDegeri",
            "NAV": "BirimPayDegeri",
            # Portföy Büyüklüğü
            "Portföy Büyüklüğü": "ToplamDeger",
            "PORTFOYBUYUKLUGU": "ToplamDeger",
            # Yatırımcı
            "Yatırımcı Sayısı": "YatirimciSayisi",
        }
        df = df.rename(columns={k: v for k, v in eslesme.items() if k in df.columns})

        # Zorunlu kolonlar
        zorunlu = ["Tarih", "BirimPayDegeri"]
        for z in zorunlu:
            if z not in df.columns:
                raise ValueError(f"Kolon bulunamadı: {z}. Dosya: {dosya_adi}")

        # FonKodu yoksa dosya adından al
        if "FonKodu" not in df.columns:
            fon_kod = dosya_adi.split(".")[0].upper()[:6]
            df["FonKodu"] = fon_kod

        # FonAdi yoksa FonKodu kullan
        if "FonAdi" not in df.columns:
            df["FonAdi"] = df["FonKodu"]

        # Tarih ayrıştır
        df["Tarih"] = pd.to_datetime(df["Tarih"], dayfirst=True, errors="coerce")
        df["BirimPayDegeri"] = pd.to_numeric(
            df["BirimPayDegeri"].astype(str).str.replace(",", ".").str.replace("₺", ""),
            errors="coerce"
        )
        df = df.dropna(subset=["Tarih", "BirimPayDegeri"])
        df = df[df["BirimPayDegeri"] > 0]

        for ek in ["ToplamDeger", "YatirimciSayisi", "FonTuru"]:
            if ek not in df.columns:
                df[ek] = np.nan

        return df[["Tarih", "FonKodu", "FonAdi", "BirimPayDegeri", "ToplamDeger", "YatirimciSayisi", "FonTuru"]]

    # ──────────────────────────────────────────────────────────────
    def demo_veri_olustur(self, fon_turleri: list = None) -> pd.DataFrame:
        """
        Gerçekçi TEFAS benzeri demo veri üretir.
        Altın fiyat hareketlerini simüle eder.
        """
        np.random.seed(42)
        gun_sayisi = 365 * 2  # 2 yıl
        tarihler   = pd.date_range(end=datetime.today(), periods=gun_sayisi, freq="B")

        parcalar = []

        # Altın bazlı gerçekçi başlangıç fiyatları (TL cinsinden)
        fon_parametreleri = {
            "ATA":  (3.50,  0.0010, 0.0002, "Ata Portföy Altın Fonu",        "Altın Fonu"),
            "GAF":  (2.80,  0.0010, 0.0002, "Garanti Altın Fonu",             "Altın Fonu"),
            "ING":  (4.20,  0.0011, 0.0003, "İNG Portföy Kıymetli Madenler", "Kıymetli Madenler"),
            "ZLT":  (1.95,  0.0009, 0.0002, "Ziraat Altın Fonu",              "Altın Fonu"),
            "YDA":  (3.10,  0.0010, 0.0002, "Yapı Kredi Altın Fonu",          "Altın Fonu"),
            "EMT1": (2.45,  0.0008, 0.0004, "TEB Emtia Fonu",                 "Emtia Fonu"),
        }

        # Ortak altın trendi (tüm fonları etkiler)
        altin_trend  = np.cumsum(np.random.normal(0.0005, 0.008, gun_sayisi))
        # Mevsimsellik bileşeni
        mevsim = 0.02 * np.sin(np.linspace(0, 4 * np.pi, gun_sayisi))

        for kod, (baslangic, drift, volatilite, ad, tur) in fon_parametreleri.items():
            # Her fonun kendine özgü ek gürültüsü
            govde_gürültu = np.random.normal(0, volatilite, gun_sayisi)
            fiyat_log     = (
                np.log(baslangic)
                + drift * np.arange(gun_sayisi)
                + 0.7 * altin_trend          # altın korelasyonu
                + mevsim                     # mevsimsellik
                + np.cumsum(govde_gürültu)   # fon özgün hareketi
            )
            fiyatlar = np.exp(fiyat_log)

            parcalar.append(pd.DataFrame({
                "Tarih":           tarihler,
                "FonKodu":         kod,
                "FonAdi":          ad,
                "BirimPayDegeri":  np.round(fiyatlar, 6),
                "ToplamDeger":     np.random.randint(50_000_000, 2_000_000_000, gun_sayisi).astype(float),
                "YatirimciSayisi": np.random.randint(500, 50_000, gun_sayisi),
                "FonTuru":         tur,
            }))

        df = pd.concat(parcalar, ignore_index=True)
        df = df.sort_values(["FonKodu", "Tarih"]).reset_index(drop=True)
        return df
