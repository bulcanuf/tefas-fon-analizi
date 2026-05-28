"""
TEFAS Fon Performans Analizi ve Tahmin Uygulaması
==================================================
Altın & Emtia Fonları odaklı ML tabanlı analiz aracı.

Kullanım:
    pip install streamlit pandas numpy scikit-learn plotly requests beautifulsoup4 prophet
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from data_loader import TEFASDataLoader
from ml_models import FonTahminModeli
from analyzer import FonAnalizci

# ─── Sayfa Ayarları ──────────────────────────────────────────────
st.set_page_config(
    page_title="TEFAS Fon Analizi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Stilleri ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa, #e8ecf0);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }
    .positive { color: #22c55e; font-weight: 700; }
    .negative { color: #ef4444; font-weight: 700; }
    .neutral  { color: #f59e0b; font-weight: 700; }
    .stAlert  { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Başlık ───────────────────────────────────────────────────────
st.markdown('<div class="main-header">📊 TEFAS Fon Analiz Merkezi</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Altın & Emtia Fonları · ML Tahmin · Performans Analizi</div>', unsafe_allow_html=True)

# ─── Kenar Çubuğu ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://www.tefas.gov.tr/Content/img/tefasLogo.png", width=180)
    st.markdown("---")
    st.markdown("### ⚙️ Veri Kaynağı")

    veri_kaynagi = st.radio(
        "Veri nasıl yüklensin?",
        ["🌐 Otomatik (TEFAS API)", "📁 CSV Yükle (Manuel)"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🎯 Fon Türü Filtresi")
    fon_turleri = st.multiselect(
        "Fon kategorisi seç:",
        ["Altın Fonu", "Emtia Fonu", "Kıymetli Madenler", "Altın Borsa Yatırım Fonu"],
        default=["Altın Fonu", "Kıymetli Madenler"]
    )

    st.markdown("---")
    st.markdown("### 📅 Tarih Aralığı")
    bitis = datetime.today()
    baslangic = bitis - timedelta(days=365)
    tarih_aralik = st.date_input(
        "Analiz dönemi:",
        value=(baslangic.date(), bitis.date()),
        min_value=datetime(2010, 1, 1).date(),
        max_value=bitis.date()
    )

    st.markdown("---")
    st.markdown("### 🤖 ML Model Ayarları")
    model_secimi = st.selectbox(
        "Tahmin modeli:",
        ["Prophet (Trend Tahmini)", "Random Forest", "Linear Regression", "Tüm Modeller (Karşılaştır)"],
        index=0
    )
    tahmin_vadesi_kisa = st.slider("Kısa vade (gün):", 7, 30, 14)
    tahmin_vadesi_orta = st.slider("Orta vade (gün):", 30, 90, 60)

    analiz_btn = st.button("🚀 Analizi Başlat", type="primary", width="stretch")

# ─── Veri Yükleme Bölümü ──────────────────────────────────────────
loader = TEFASDataLoader()

if "🌐 Otomatik" in veri_kaynagi:
    if analiz_btn or "df" not in st.session_state:
        with st.spinner("TEFAS'tan veriler çekiliyor..."):
            try:
                df = loader.tefas_api_cek(fon_turleri, tarih_aralik)
                st.session_state["df"] = df
                st.success(f"✅ {len(df['FonKodu'].unique())} fon başarıyla yüklendi.")
            except Exception as e:
                st.error(f"API hatası: {e}")
                st.info("Manuel CSV yüklemeyi deneyin.")
                st.stop()
else:
    st.markdown("### 📁 CSV Dosyası Yükle")
    st.markdown("""
    TEFAS'tan CSV indirmek için:
    1. [tefas.gov.tr](https://www.tefas.gov.tr) → Fon Karşılaştırma
    2. İstediğiniz fonları seçin
    3. "Excel/CSV İndir" butonuna tıklayın
    """)
    yuklenen = st.file_uploader(
        "TEFAS CSV dosyasını sürükleyin",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )
    if yuklenen:
        df = loader.csv_yukle(yuklenen)
        st.session_state["df"] = df
        st.success(f"✅ {len(df['FonKodu'].unique())} fon yüklendi.")
    elif "df" not in st.session_state:
        # Demo mod
        st.info("ℹ️ Dosya yüklenmedi — demo verilerle çalışılıyor.")
        df = loader.demo_veri_olustur(fon_turleri)
        st.session_state["df"] = df

# ─── Veri Kontrolü ────────────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("Lütfen önce veri yükleyin veya analizi başlatın.")
    st.stop()

df = st.session_state["df"]

# ─── Fon Seçimi ───────────────────────────────────────────────────
st.markdown("---")
col_sec1, col_sec2 = st.columns([3, 1])
with col_sec1:
    fon_listesi = sorted(df["FonKodu"].unique().tolist())
    secili_fonlar = st.multiselect(
        "🔍 Analiz edilecek fonları seçin:",
        fon_listesi,
        default=fon_listesi[:min(5, len(fon_listesi))],
        help="Birden fazla fon seçerek karşılaştırma yapabilirsiniz."
    )
with col_sec2:
    benchmark = st.selectbox("📌 Benchmark:", ["Altın (TL/g)", "BIST100", "USD/TRY", "Yok"])

if not secili_fonlar:
    st.warning("En az bir fon seçin.")
    st.stop()

df_secili = df[df["FonKodu"].isin(secili_fonlar)].copy()
analizci = FonAnalizci(df_secili)

# ─── Tab Yapısı ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performans", "🤖 ML Tahmin", "⚖️ Risk Analizi", "📊 İstatistikler", "🏆 Sıralama"
])

# ────────────────────────────────────────────────────────────────────
# TAB 1 — PERFORMANS
# ────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Fiyat Performansı")

    # Özet metrikler
    cols = st.columns(len(secili_fonlar[:5]))
    for i, fon in enumerate(secili_fonlar[:5]):
        fon_df = df_secili[df_secili["FonKodu"] == fon]
        getiri = analizci.toplam_getiri(fon)
        son_fiyat = fon_df["BirimPayDegeri"].iloc[-1]
        fon_adi = fon_df["FonAdi"].iloc[0] if "FonAdi" in fon_df.columns else fon
        with cols[i]:
            renk = "positive" if getiri > 0 else "negative"
            st.metric(
                label=f"**{fon}**",
                value=f"₺{son_fiyat:.4f}",
                delta=f"{getiri:+.2f}%"
            )

    st.markdown("---")

    # Normalize fiyat grafiği
    fig_norm = go.Figure()
    renkler = px.colors.qualitative.Set2
    for i, fon in enumerate(secili_fonlar):
        fon_df = df_secili[df_secili["FonKodu"] == fon].sort_values("Tarih")
        ilk = fon_df["BirimPayDegeri"].iloc[0]
        normalize = (fon_df["BirimPayDegeri"] / ilk - 1) * 100
        fig_norm.add_trace(go.Scatter(
            x=fon_df["Tarih"],
            y=normalize,
            name=fon,
            line=dict(color=renkler[i % len(renkler)], width=2),
            hovertemplate=f"<b>{fon}</b><br>Tarih: %{{x|%d.%m.%Y}}<br>Getiri: %{{y:.2f}}%<extra></extra>"
        ))
    fig_norm.update_layout(
        title="Normalize Getiri (Başlangıç = 0%)",
        xaxis_title="Tarih", yaxis_title="Kümülatif Getiri (%)",
        hovermode="x unified", height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.15)
    )
    fig_norm.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_norm, width="stretch")

    # Aylık getiri ısı haritası
    st.markdown("### 📅 Aylık Getiri Isı Haritası")
    fon_isi = st.selectbox("Fon seçin:", secili_fonlar, key="isi")
    heatmap_data = analizci.aylik_getiri_matrisi(fon_isi)
    if heatmap_data is not None:
        fig_heat = px.imshow(
            heatmap_data,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            text_auto=".1f",
            labels=dict(color="Getiri (%)"),
            title=f"{fon_isi} — Aylık Getiri Matrisi (%)"
        )
        fig_heat.update_layout(height=300)
        st.plotly_chart(fig_heat, width="stretch")

# ────────────────────────────────────────────────────────────────────
# TAB 2 — ML TAHMİN
# ────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🤖 Makine Öğrenmesi ile Fiyat Tahmini")
    st.info("""
    **Modeller:** Prophet (trend+mevsimsellik), Random Forest (pattern tanıma), Linear Regression (baseline)  
    ⚠️ Tahminler yatırım tavsiyesi değildir. Geçmiş performans gelecek getiriyi garanti etmez.
    """)

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        tahmin_fon = st.selectbox("Tahmin yapılacak fon:", secili_fonlar, key="tahmin_fon")
    with col_t2:
        tahmin_modeli_sec = st.selectbox(
            "Model:", ["Prophet", "Random Forest", "Linear Regression", "Ensemble (Ortalama)"],
            key="model_sec"
        )

    if st.button("🔮 Tahmin Et", type="primary"):
        fon_df = df_secili[df_secili["FonKodu"] == tahmin_fon].sort_values("Tarih")

        with st.spinner(f"{tahmin_fon} için {tahmin_modeli_sec} modeli çalıştırılıyor..."):
            model = FonTahminModeli(fon_df)

            col_kisa, col_orta = st.columns(2)

            for vade, gun, col in [
                ("Kısa Vade", tahmin_vadesi_kisa, col_kisa),
                ("Orta Vade", tahmin_vadesi_orta, col_orta)
            ]:
                tahmin_df, metrikler = model.tahmin_et(tahmin_modeli_sec, gun)

                with col:
                    st.markdown(f"#### {vade} ({gun} gün)")

                    # Metrikler
                    son = fon_df["BirimPayDegeri"].iloc[-1]
                    tahmini_son = tahmin_df["yhat"].iloc[-1]
                    degisim = (tahmini_son / son - 1) * 100
                    renk = "🟢" if degisim > 0 else "🔴"

                    st.metric(
                        f"{renk} Tahmini Fiyat ({gun}. gün)",
                        f"₺{tahmini_son:.4f}",
                        delta=f"{degisim:+.2f}%"
                    )
                    if metrikler:
                        st.caption(f"Model MAPE: {metrikler.get('mape', 'N/A')}")

                    # Tahmin grafiği
                    fig_t = go.Figure()
                    fig_t.add_trace(go.Scatter(
                        x=fon_df["Tarih"].tail(60),
                        y=fon_df["BirimPayDegeri"].tail(60),
                        name="Gerçek", line=dict(color="#334155", width=2)
                    ))
                    fig_t.add_trace(go.Scatter(
                        x=tahmin_df["ds"], y=tahmin_df["yhat"],
                        name="Tahmin", line=dict(color="#6366f1", width=2, dash="dash")
                    ))
                    if "yhat_lower" in tahmin_df.columns:
                        fig_t.add_trace(go.Scatter(
                            x=pd.concat([tahmin_df["ds"], tahmin_df["ds"][::-1]]),
                            y=pd.concat([tahmin_df["yhat_upper"], tahmin_df["yhat_lower"][::-1]]),
                            fill="toself", fillcolor="rgba(99,102,241,0.1)",
                            line=dict(color="rgba(255,255,255,0)"),
                            name="Güven Aralığı"
                        ))
                    fig_t.update_layout(
                        height=300, showlegend=True,
                        plot_bgcolor="white", paper_bgcolor="white",
                        margin=dict(l=0, r=0, t=10, b=0),
                        legend=dict(orientation="h", y=-0.2)
                    )
                    st.plotly_chart(fig_t, width="stretch")

# ────────────────────────────────────────────────────────────────────
# TAB 3 — RİSK ANALİZİ
# ────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### ⚖️ Risk Metrikleri")

    risk_df = analizci.risk_metrikleri_tumu(secili_fonlar)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        # Risk-Getiri scatter
        fig_rg = px.scatter(
            risk_df, x="Volatilite (%)", y="Getiri (%)",
            text="FonKodu", color="Sharpe",
            color_continuous_scale="RdYlGn",
            title="Risk-Getiri Grafiği",
            labels={"Volatilite (%)": "Yıllık Volatilite (%)", "Getiri (%)": "Dönem Getirisi (%)"}
        )
        fig_rg.update_traces(textposition="top center", marker_size=12)
        fig_rg.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_rg, width="stretch")

    with col_r2:
        # Volatilite çubuk grafik
        fig_vol = px.bar(
            risk_df.sort_values("Volatilite (%)"),
            x="FonKodu", y="Volatilite (%)",
            color="Volatilite (%)", color_continuous_scale="YlOrRd",
            title="Fon Volatiliteleri (%)"
        )
        fig_vol.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_vol, width="stretch")

    # Risk tablosu
    st.markdown("### 📋 Risk Metrikleri Tablosu")
    st.dataframe(
        risk_df.style
            .background_gradient(subset=["Sharpe"], cmap="RdYlGn")
            .background_gradient(subset=["Max Drawdown (%)"], cmap="RdYlGn_r")
            .format({
                "Getiri (%)": "{:.2f}%",
                "Volatilite (%)": "{:.2f}%",
                "Sharpe": "{:.3f}",
                "Max Drawdown (%)": "{:.2f}%",
                "VaR 95% (%)": "{:.2f}%"
            }),
        width="stretch", height=300
    )

# ────────────────────────────────────────────────────────────────────
# TAB 4 — İSTATİSTİKLER
# ────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 📊 Detaylı İstatistikler")

    istat_fon = st.selectbox("Fon seçin:", secili_fonlar, key="istat_fon")
    fon_df_istat = df_secili[df_secili["FonKodu"] == istat_fon].sort_values("Tarih")

    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    getiriler = fon_df_istat["BirimPayDegeri"].pct_change().dropna() * 100

    col_i1.metric("Ortalama Günlük Getiri", f"{getiriler.mean():.4f}%")
    col_i2.metric("Std. Sapma", f"{getiriler.std():.4f}%")
    col_i3.metric("Çarpıklık", f"{getiriler.skew():.3f}")
    col_i4.metric("Basıklık", f"{getiriler.kurtosis():.3f}")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_hist = px.histogram(
            getiriler, nbins=50, title="Günlük Getiri Dağılımı",
            labels={"value": "Günlük Getiri (%)"},
            color_discrete_sequence=["#6366f1"]
        )
        fig_hist.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_hist, width="stretch")

    with col_g2:
        # Kümülatif getiri
        kumulatif = (1 + getiriler / 100).cumprod() - 1
        fig_kum = px.area(
            x=fon_df_istat["Tarih"].iloc[1:], y=kumulatif * 100,
            title="Kümülatif Getiri (%)",
            labels={"x": "Tarih", "y": "Kümülatif Getiri (%)"},
            color_discrete_sequence=["#22c55e"]
        )
        fig_kum.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_kum, width="stretch")

    # Korelasyon matrisi (birden fazla fon varsa)
    if len(secili_fonlar) > 1:
        st.markdown("### 🔗 Fon Korelasyon Matrisi")
        korel = analizci.korelasyon_matrisi(secili_fonlar)
        if korel is not None:
            fig_korel = px.imshow(
                korel, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                title="Getiri Korelasyonları"
            )
            fig_korel.update_layout(height=400)
            st.plotly_chart(fig_korel, width="stretch")

# ────────────────────────────────────────────────────────────────────
# TAB 5 — SIRALAMA
# ────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 🏆 Fon Performans Sıralaması")

    siralama_kriteri = st.selectbox(
        "Sıralama kriteri:",
        ["Dönem Getirisi (%)", "Sharpe Oranı", "Volatilite (Düşük→Yüksek)", "Max Drawdown"]
    )

    siralama_df = analizci.siralama_tablosu(secili_fonlar, siralama_kriteri)
    if siralama_df is not None:
        # Madalya emojisi ekle
        siralama_df.insert(0, "Sıra", ["🥇", "🥈", "🥉"] + [f"{i+4}." for i in range(max(0, len(siralama_df)-3))])

        st.dataframe(
            siralama_df.style
                .background_gradient(subset=["Dönem Getirisi (%)"], cmap="RdYlGn")
                .format({
                    "Dönem Getirisi (%)": "{:.2f}%",
                    "Sharpe": "{:.3f}",
                    "Volatilite (%)": "{:.2f}%",
                    "Max Drawdown (%)": "{:.2f}%"
                }),
            width="stretch",
            hide_index=True,
            height=400
        )

        # En iyi fon özeti
        en_iyi = siralama_df.iloc[0]
        st.success(f"🏆 **En iyi fon:** {en_iyi.get('FonKodu', 'N/A')} — {en_iyi.get('Dönem Getirisi (%)', 0):.2f}% getiri, {en_iyi.get('Sharpe', 0):.3f} Sharpe oranı")

# ─── Alt Bilgi ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.8rem;">
TEFAS Fon Analiz Merkezi · Veriler: tefas.gov.tr · Bu araç yatırım tavsiyesi vermez.
</div>
""", unsafe_allow_html=True)
