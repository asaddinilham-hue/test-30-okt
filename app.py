
from pathlib import Path
from io import BytesIO
import re

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st

# =====================================================
# KONFIGURASI
# =====================================================
st.set_page_config(
    page_title="Coil Quality Dashboard",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Coil Production & Quality Dashboard")
st.caption(
    "Interactive dashboard untuk monitoring coil, proses produksi, "
    "target vs actual, dan kualitas produk."
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Kolom yang umum digunakan sebagai identitas/tooltip
DETAIL_COLUMNS = [
    "INLET_COIL_NO",
    "OTLET_COIL_NO",
    "JSG_DESI_NM",
    "UPDAT_DT",
    "SHEAR_CUT_DT",
    "WORK_DT",
    "RSLT_CD",
    "COIL_WT",
    "INLET_COIL_TH",
    "INLET_COIL_WH",
    "NO1_TRG_GRADE_CD",
    "TOP_SRFCE_GRADE_CD",
    "SHAPE_GRADE_CD",
    "BOTTM_SRFCE_GRADE_CD",
    "NO1_DEFCT_CD",
    "NO1_DEFCT_GRADE_CD",
    "NO2_DEFCT_CD",
    "NO2_DEFCT_GRADE_CD",
    "NO3_DEFCT_CD",
    "NO3_DEFCT_GRADE_CD",
    "Thickness group",
]

DATE_CANDIDATES = [
    "Date", "WORK_DT", "UPDAT_DT", "SHEAR_CUT_DT"
]


# =====================================================
# FUNGSI BACA FILE
# =====================================================
@st.cache_data(show_spinner=False)
def read_excel_file(file_bytes, file_name, sheet_name, header_row):
    suffix = Path(file_name).suffix.lower()
    bio = BytesIO(file_bytes)

    if suffix == ".csv":
        return pd.read_csv(bio)

    return pd.read_excel(
        bio,
        sheet_name=sheet_name,
        header=header_row - 1,
    )


def clean_column_names(df):
    df = df.copy()
    df.columns = [
        re.sub("[\u200b\u200c\u200d\ufeff\xa0]", "", str(c)).strip()
        for c in df.columns
    ]
    return df


def parse_date_column(series, column_name):
    """Menangani Date, YYYYMMDD, serta timestamp umum."""
    if column_name == "WORK_DT":
        text = series.astype("string").str.strip()
        result = pd.to_datetime(
            text, format="%Y%m%d", errors="coerce"
        )
        fallback = result.isna() & text.notna()
        if fallback.any():
            result.loc[fallback] = pd.to_datetime(
                text.loc[fallback], errors="coerce"
            )
        return result

    if column_name == "SHEAR_CUT_DT":
        text = series.astype("string").str.strip()
        # Format contoh: 20250108080140
        result = pd.to_datetime(
            text, format="%Y%m%d%H%M%S", errors="coerce"
        )
        fallback = result.isna() & text.notna()
        if fallback.any():
            result.loc[fallback] = pd.to_datetime(
                text.loc[fallback], errors="coerce"
            )
        return result

    # UPDAT_DT dan Date biasanya berupa tanggal yang bisa dibaca pandas
    return pd.to_datetime(series, errors="coerce", format="mixed")


def prepare_data(raw_df):
    df = clean_column_names(raw_df)

    # Buang baris kosong sepenuhnya
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(how="all").copy()

    # Bersihkan karakter tersembunyi dari kolom teks
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = (
            df[col].astype("string")
            .str.replace(
                "[\u200b\u200c\u200d\ufeff\xa0]", "", regex=True
            )
            .str.strip()
        )
        df[col] = df[col].replace("", pd.NA)

    # Buat kolom tanggal analisis
    date_source = next(
        (c for c in DATE_CANDIDATES if c in df.columns),
        None,
    )

    if date_source:
        df["_ANALYSIS_DATE"] = parse_date_column(
            df[date_source], date_source
        )
    else:
        df["_ANALYSIS_DATE"] = pd.NaT

    # Kolom yang pasti berbentuk angka
    numeric_candidates = [
        "INLET_COIL_TH", "INLET_COIL_WH", "INLET_COIL_LG",
        "COIL_WT", "ACT_WK_TH", "MIN_CE_SP", "AVE_CE_SP",
        "MAX_CE_SP", "Target1", "Target2", "Target3",
        "Actual1", "Actual 2", "Actual3",
        "CGL UP", "Cooling Rate",
        "SKP_R_AVE_EXTEN_PC", "SKP_R_ROLL_LOAD",
        "Rolling Force (x10 kN/mm)",
        "SKP_R_DLV_TENSN_TTL", "SKP_R_ENT_TENSN_TTL",
        "Year", "Month", "Day",
        "YP Tail", "TS Tail", "EL Tail", "BH Tail",
        "YP LL", "YP UL", "TS LL", "TS UL",
        "EL LL", "EL UL", "BH LL", "BH UL",
        "DEVIDER", "Sample",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coba deteksi kolom numerik tambahan yang tersimpan sebagai teks.
    # Kolom kode/identitas tidak otomatis dikonversi.
    protected = {
        "INLET_COIL_NO", "OTLET_COIL_NO", "JSG_DESI_NM",
        "RSLT_CD", "NO1_TRG_GRADE_CD", "TOP_SRFCE_GRADE_CD",
        "SHAPE_GRADE_CD", "BOTTM_SRFCE_GRADE_CD",
        "NO1_DEFCT_CD", "NO1_DEFCT_GRADE_CD",
        "NO2_DEFCT_CD", "NO2_DEFCT_GRADE_CD",
        "NO3_DEFCT_CD", "NO3_DEFCT_GRADE_CD",
        "REPET_ORDER_NO", "C00_CHEM_CD",
        "Thickness group", "Date", "UPDAT_DT",
        "WORK_DT", "SHEAR_CUT_DT",
    }

    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col in protected:
            continue

        converted = pd.to_numeric(
            df[col].astype("string").str.replace(",", ".", regex=False),
            errors="coerce",
        )
        non_empty = df[col].notna().sum()

        if non_empty > 0 and converted.notna().sum() / non_empty >= 0.85:
            df[col] = converted

    # Kolom tampilan target-actual
    target_actual_pairs = [
        ("Target1", "Actual1", "Target 1 vs Actual 1"),
        ("Target2", "Actual 2", "Target 2 vs Actual 2"),
        ("Target3", "Actual3", "Target 3 vs Actual 3"),
    ]

    return df, date_source, target_actual_pairs


# =====================================================
# SIDEBAR: SUMBER DATA
# =====================================================
st.sidebar.header("⚙️ Pengaturan")

source_mode = st.sidebar.radio(
    "Sumber data",
    ["Upload file", "Baca dari direktori"],
)

file_bytes = None
file_name = None

if source_mode == "Upload file":
    uploaded = st.sidebar.file_uploader(
        "Upload Excel atau CSV",
        type=["xlsx", "xls", "xlsm", "csv"],
    )
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_name = uploaded.name

else:
    directory_text = st.sidebar.text_input(
        "Direktori file",
        value=str(Path.cwd()),
        help="Direktori dibaca pada komputer/server tempat Streamlit berjalan.",
    )
    directory = Path(directory_text).expanduser()

    if directory.is_dir():
        available_files = sorted([
            p for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".xlsx", ".xls", ".xlsm", ".csv"}
        ])

        if available_files:
            selected_file = st.sidebar.selectbox(
                "Pilih file",
                available_files,
                format_func=lambda p: p.name,
            )
            file_bytes = selected_file.read_bytes()
            file_name = selected_file.name
        else:
            st.sidebar.warning("Tidak ada file Excel atau CSV.")
    else:
        st.sidebar.error("Direktori tidak ditemukan.")


# =====================================================
# LOAD DATA
# =====================================================
if file_bytes is None or file_name is None:
    st.info("Upload file atau pilih file dari direktori untuk memulai.")
    st.stop()

try:
    if Path(file_name).suffix.lower() == ".csv":
        raw_df = read_excel_file(
            file_bytes, file_name, None, 1
        )
    else:
        workbook = pd.ExcelFile(BytesIO(file_bytes))
        sheet_name = st.sidebar.selectbox(
            "Pilih sheet",
            workbook.sheet_names,
        )
        header_row = st.sidebar.number_input(
            "Baris header Excel",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
        )
        raw_df = read_excel_file(
            file_bytes, file_name, sheet_name, int(header_row)
        )

    df, date_source, target_actual_pairs = prepare_data(raw_df)

except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

if df.empty:
    st.warning("File tidak memiliki baris data.")
    st.stop()

st.caption(f"File aktif: {file_name} | {len(df):,} baris | {len(df.columns)} kolom")


# =====================================================
# FILTER DATA
# =====================================================
st.sidebar.divider()
st.sidebar.subheader("🔎 Filter")

filtered = df.copy()

if "_ANALYSIS_DATE" in filtered.columns and filtered["_ANALYSIS_DATE"].notna().any():
    valid_dates = filtered["_ANALYSIS_DATE"].dropna()

    date_range = st.sidebar.date_input(
        "Rentang tanggal",
        value=(valid_dates.min().date(), valid_dates.max().date()),
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            filtered["_ANALYSIS_DATE"].isna()
            | (
                filtered["_ANALYSIS_DATE"].dt.date.ge(start_date)
                & filtered["_ANALYSIS_DATE"].dt.date.le(end_date)
            )
        ]

# Filter kategori
filter_options = [
    "INLET_COIL_NO",
    "OTLET_COIL_NO",
    "NO1_TRG_GRADE_CD",
    "TOP_SRFCE_GRADE_CD",
    "SHAPE_GRADE_CD",
    "BOTTM_SRFCE_GRADE_CD",
    "RSLT_CD",
    "Thickness group",
    "CAL/CGL",
]

for col in filter_options:
    if col in filtered.columns:
        options = sorted(
            filtered[col].dropna().astype(str).unique().tolist()
        )
        if options and len(options) <= 200:
            selected = st.sidebar.multiselect(
                col,
                options,
                default=[],
                key=f"filter_{col}",
            )
            if selected:
                filtered = filtered[
                    filtered[col].astype(str).isin(selected)
                ]

if filtered.empty:
    st.warning("Tidak ada data yang cocok dengan filter.")
    st.stop()


# =====================================================
# KPI
# =====================================================
st.subheader("Ringkasan Produksi")

k1, k2, k3, k4 = st.columns(4)

if "INLET_COIL_NO" in filtered.columns:
    total_coils = filtered["INLET_COIL_NO"].nunique()
else:
    total_coils = len(filtered)

k1.metric("Total Coil Unik", f"{total_coils:,}")
k2.metric("Jumlah Record", f"{len(filtered):,}")

if "COIL_WT" in filtered.columns:
    avg_weight = filtered["COIL_WT"].mean()
    k3.metric(
        "Rata-rata Coil Weight",
        f"{avg_weight:,.2f}" if pd.notna(avg_weight) else "-",
    )
else:
    k3.metric("Rata-rata Coil Weight", "-")

if "RSLT_CD" in filtered.columns:
    result_count = filtered["RSLT_CD"].nunique()
    k4.metric("Kategori Result", f"{result_count:,}")
else:
    k4.metric("Kategori Result", "-")


# =====================================================
# PILIH KOLOM NUMERIK DAN DETAIL TOOLTIP
# =====================================================
numeric_cols = [
    c for c in filtered.select_dtypes(include="number").columns
    if not c.startswith("_")
]

preferred_metrics = [
    "COIL_WT", "INLET_COIL_TH", "INLET_COIL_WH",
    "ACT_WK_TH", "Target1", "Target2", "Target3",
    "Actual1", "Actual 2", "Actual3",
    "AVE_CE_SP", "MIN_CE_SP", "MAX_CE_SP",
    "Cooling Rate", "Rolling Force (x10 kN/mm)",
]

preferred_metrics = [
    c for c in preferred_metrics if c in numeric_cols
]

if not preferred_metrics:
    preferred_metrics = numeric_cols

hover_defaults = [
    c for c in DETAIL_COLUMNS if c in filtered.columns
]
hover_defaults += [
    c for c in [
        "Target1", "Target2", "Target3",
        "Actual1", "Actual 2", "Actual3",
        "AVE_CE_SP", "Cooling Rate",
    ]
    if c in filtered.columns and c not in hover_defaults
]

with st.expander("⚙️ Pengaturan tooltip grafik", expanded=False):
    hover_columns = st.multiselect(
        "Pilih informasi yang muncul saat kursor diarahkan ke titik grafik",
        options=[c for c in filtered.columns if not c.startswith("_")],
        default=hover_defaults,
    )


# =====================================================
# TAB DASHBOARD
# =====================================================
tab_trend, tab_target, tab_quality, tab_data = st.tabs(
    [
        "📈 Tren Proses",
        "🎯 Target vs Actual",
        "🛡️ Quality & Defect",
        "📋 Data",
    ]
)

# =====================================================
# TAB 1: LINE CHART INTERAKTIF
# =====================================================
with tab_trend:

    st.subheader("Tren Parameter Proses")

    if not numeric_cols:

        st.warning(
            "Tidak ada kolom numerik untuk divisualisasikan."
        )

    else:

        # =============================================
        # PEMILIHAN SUMBU X
        # =============================================

        x_options = [
            c for c in filtered.columns
            if c in {
                "_ANALYSIS_DATE",
                "INLET_COIL_NO",
                "OTLET_COIL_NO",
                "Date",
                "WORK_DT",
                "UPDAT_DT",
            }
            or c in numeric_cols
        ]

        # UPDAT_DT menjadi pilihan default
        if "UPDAT_DT" in x_options:
            x_default = "UPDAT_DT"
        elif "_ANALYSIS_DATE" in x_options:
            x_default = "_ANALYSIS_DATE"
        else:
            x_default = x_options[0]

        c1, c2 = st.columns(2)

        x_col = c1.selectbox(
            "Sumbu X",
            x_options,
            index=x_options.index(x_default),
            format_func=lambda x: (
                "Tanggal analisis"
                if x == "_ANALYSIS_DATE"
                else x
            ),
            key="trend_x_col"
        )

        # =============================================
        # MULTIPLE PARAMETER Y
        # =============================================

        y_cols = c2.multiselect(
            "Parameter Y (Pilih Multiple Parameter)",
            options=numeric_cols,
            default=(
                [preferred_metrics[0]]
                if preferred_metrics
                else [numeric_cols[0]]
            ),
            help="Pilih satu atau beberapa parameter untuk dibuatkan grafik.",
            key="trend_y_cols"
        )

        # =============================================
        # PENGATURAN WARNA
        # =============================================

        color_options = ["(Tidak ada)"] + [
            c for c in [
                "NO1_TRG_GRADE_CD",
                "TOP_SRFCE_GRADE_CD",
                "SHAPE_GRADE_CD",
                "RSLT_CD",
                "Thickness group",
                "CAL/CGL",
            ]
            if c in filtered.columns
        ]

        color_col = st.selectbox(
            "Warna berdasarkan",
            color_options,
            key="trend_color"
        )

        color_arg = (
            None
            if color_col == "(Tidak ada)"
            else color_col
        )

        # =============================================
        # JENIS GRAFIK
        # =============================================

        chart_type = st.radio(
            "Jenis grafik",
            ["Line", "Scatter", "Bar"],
            horizontal=True,
            key="trend_chart_type"
        )

        st.markdown("---")

        # =============================================
        # VALIDASI PARAMETER Y
        # =============================================

        if not y_cols:

            st.info(
                "Silakan pilih minimal satu Parameter Y."
            )

        else:

            st.subheader(
                f"Grafik Tren ({len(y_cols)} Parameter)"
            )

            date_columns = [
                "_ANALYSIS_DATE",
                "UPDAT_DT",
                "Date",
                "WORK_DT",
                "SHEAR_CUT_DT",
            ]

            # =========================================
            # LOOP SETIAP PARAMETER Y
            # =========================================

            for idx, y_col in enumerate(y_cols):

                st.markdown("---")

                st.markdown(
                    f"### Grafik {idx + 1}: {y_col}"
                )

                # =====================================
                # PERSIAPAN DATA
                # =====================================

                metric_df = filtered.copy()

                # Konversi parameter Y
                metric_df[y_col] = pd.to_numeric(
                    metric_df[y_col],
                    errors="coerce"
                )

                # Konversi sumbu X jika tanggal
                if x_col in date_columns:

                    metric_df[x_col] = pd.to_datetime(
                        metric_df[x_col],
                        errors="coerce"
                    )

                # Hapus data kosong pada X dan Y
                metric_df = metric_df.dropna(
                    subset=[x_col, y_col]
                )

                # Urutkan berdasarkan waktu
                if x_col in date_columns:

                    metric_df = metric_df.sort_values(
                        x_col
                    ).reset_index(drop=True)

                    x_label = (
                        "UPDAT_DT (Tanggal dan Waktu)"
                        if x_col == "UPDAT_DT"
                        else "Tanggal dan Waktu"
                    )

                else:

                    x_label = x_col

                # Validasi data
                if metric_df.empty:

                    st.warning(
                        f"Tidak ada data valid untuk {y_col}."
                    )

                    continue

                # =====================================
                # PENGATURAN UPPER LIMIT DAN LOWER LIMIT
                # =====================================

                if y_col == "ANLNG_S_TP_HSD_TP_STRIP":

                    default_upper = 820.0
                    default_lower = 780.0

                else:

                    default_upper = float(
                        metric_df[y_col].max()
                    )

                    default_lower = float(
                        metric_df[y_col].min()
                    )

                with st.expander(
                    f"⚙️ Pengaturan Batas: {y_col}",
                    expanded=True
                ):

                    enable_limit = st.checkbox(
                        "Aktifkan Upper Limit dan Lower Limit",
                        value=True,
                        key=f"enable_limit_{idx}_{y_col}"
                    )

                    c_upper, c_lower = st.columns(2)

                    upper_limit = c_upper.number_input(
                        "Upper Limit (Batas Atas)",
                        value=default_upper,
                        step=1.0,
                        format="%.3f",
                        key=f"upper_limit_{idx}_{y_col}"
                    )

                    lower_limit = c_lower.number_input(
                        "Lower Limit (Batas Bawah)",
                        value=default_lower,
                        step=1.0,
                        format="%.3f",
                        key=f"lower_limit_{idx}_{y_col}"
                    )

                # =====================================
                # VALIDASI BATAS
                # =====================================

                if (
                    enable_limit
                    and upper_limit <= lower_limit
                ):

                    st.error(
                        f"Upper Limit harus lebih besar "
                        f"dari Lower Limit untuk {y_col}."
                    )

                    continue

                # =====================================
                # DATA TOOLTIP
                # =====================================

                valid_hover_columns = [
                    c for c in hover_columns
                    if c in metric_df.columns
                    and c != x_col
                    and c != y_col
                ]

                # Pastikan identitas coil ditampilkan
                for c in [
                    "INLET_COIL_NO",
                    "OTLET_COIL_NO",
                    "UPDAT_DT",
                ]:

                    if (
                        c in metric_df.columns
                        and c != x_col
                        and c not in valid_hover_columns
                    ):

                        valid_hover_columns.append(c)

                # =====================================
                # PEMBUATAN GRAFIK
                # =====================================

                if chart_type == "Line":

                    fig = px.line(
                        metric_df,
                        x=x_col,
                        y=y_col,
                        color=color_arg,
                        markers=True,
                        hover_data=valid_hover_columns,
                        title=f"Tren {y_col}"
                    )

                elif chart_type == "Scatter":

                    fig = px.scatter(
                        metric_df,
                        x=x_col,
                        y=y_col,
                        color=color_arg,
                        hover_data=valid_hover_columns,
                        title=f"Sebaran {y_col}"
                    )

                else:

                    fig = px.bar(
                        metric_df,
                        x=x_col,
                        y=y_col,
                        color=color_arg,
                        hover_data=valid_hover_columns,
                        title=f"{y_col} berdasarkan {x_label}"
                    )

                # =====================================
                # FORMAT SUMBU X
                # =====================================

                if x_col in date_columns:

                    fig.update_xaxes(
                        type="date",
                        title=x_label,
                        tickformat="%d-%m-%Y<br>%H:%M:%S",
                        showgrid=True,
                        rangeslider_visible=True
                    )

                else:

                    fig.update_xaxes(
                        title=x_label,
                        showgrid=True
                    )

                # =====================================
                # GARIS UPPER LIMIT DAN LOWER LIMIT
                # =====================================

                if enable_limit:

                    fig.add_hline(
                        y=upper_limit,
                        line_dash="dash",
                        line_color="red",
                        line_width=2,
                        annotation_text=(
                            f"Upper Limit = {upper_limit:g}"
                        ),
                        annotation_position="top right"
                    )

                    fig.add_hline(
                        y=lower_limit,
                        line_dash="dash",
                        line_color="red",
                        line_width=2,
                        annotation_text=(
                            f"Lower Limit = {lower_limit:g}"
                        ),
                        annotation_position="bottom right"
                    )

                # =====================================
                # LAYOUT GRAFIK
                # =====================================

                fig.update_layout(
                    height=550,
                    hovermode="closest",
                    xaxis_title=x_label,
                    yaxis_title=y_col,
                    legend_title_text=(
                        color_col if color_arg else ""
                    ),
                    margin=dict(
                        l=50,
                        r=40,
                        t=70,
                        b=80
                    )
                )

                fig.update_traces(
                    hoverlabel=dict(
                        namelength=-1
                    )
                )

                # =====================================
                # TAMPILKAN GRAFIK
                # =====================================

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"process_chart_{idx}_{y_col}"
                )

                # =====================================
                # RINGKASAN STATISTIK
                # =====================================

                mean_value = metric_df[y_col].mean()
                max_value = metric_df[y_col].max()
                min_value = metric_df[y_col].min()

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    f"Rata-rata {y_col}",
                    f"{mean_value:,.3f}"
                )

                m2.metric(
                    "Nilai Maksimum",
                    f"{max_value:,.3f}"
                )

                m3.metric(
                    "Nilai Minimum",
                    f"{min_value:,.3f}"
                )

                st.caption(
                    "Arahkan kursor ke titik grafik untuk melihat "
                    "tanggal, waktu, nomor coil, dan nilai parameter. "
                    "Gunakan toolbar untuk zoom, pan, dan mengunduh grafik."
                )

# =====================================================
# TAB 2: TARGET VS ACTUAL
# =====================================================
with tab_target:
    st.subheader("Perbandingan Target dan Actual")

    pair_options = [
        pair for pair in target_actual_pairs
        if pair[0] in filtered.columns and pair[1] in filtered.columns
    ]

    if pair_options:
        selected_pair_name = st.selectbox(
            "Pilih pasangan parameter",
            [p[2] for p in pair_options],
        )

        target_col, actual_col, _ = next(
            p for p in pair_options if p[2] == selected_pair_name
        )

        compare_df = filtered.copy()

        if "_ANALYSIS_DATE" in compare_df.columns and compare_df["_ANALYSIS_DATE"].notna().any():
            x_compare = "_ANALYSIS_DATE"
            compare_df = compare_df.dropna(subset=[x_compare])
        elif "INLET_COIL_NO" in compare_df.columns:
            x_compare = "INLET_COIL_NO"
        else:
            x_compare = None

        if x_compare:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=compare_df[x_compare],
                    y=compare_df[target_col],
                    mode="lines+markers",
                    name=target_col,
                    customdata=compare_df[
                        [c for c in hover_columns if c in compare_df.columns]
                    ].astype(str).to_numpy(),
                    hovertemplate=(
                        "<b>Target</b><br>"
                        "X: %{x}<br>"
                        "Nilai: %{y}<extra></extra>"
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=compare_df[x_compare],
                    y=compare_df[actual_col],
                    mode="lines+markers",
                    name=actual_col,
                    hovertemplate=(
                        "<b>Actual</b><br>"
                        "X: %{x}<br>"
                        "Nilai: %{y}<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                title=f"{target_col} vs {actual_col}",
                xaxis_title=x_compare,
                yaxis_title="Nilai",
                height=500,
                hovermode="x unified",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="target_actual_chart",
            )

        target_mean = filtered[target_col].mean()
        actual_mean = filtered[actual_col].mean()

        m1, m2, m3 = st.columns(3)
        m1.metric(
            f"Rata-rata {target_col}",
            f"{target_mean:,.3f}" if pd.notna(target_mean) else "-",
        )
        m2.metric(
            f"Rata-rata {actual_col}",
            f"{actual_mean:,.3f}" if pd.notna(actual_mean) else "-",
        )

        if pd.notna(target_mean) and pd.notna(actual_mean):
            difference = actual_mean - target_mean
            m3.metric(
                "Selisih Actual - Target",
                f"{difference:,.3f}",
            )
    else:
        st.info(
            "Kolom Target1/Actual1, Target2/Actual 2, "
            "atau Target3/Actual3 tidak ditemukan."
        )

# =====================================================
# GRAFIK KHUSUS: ANLNG_S_TP_HSD_TP_STRIP
# =====================================================
st.markdown("---")
st.subheader("Monitoring ANLNG_S_TP_HSD_TP_STRIP")

required_cols = [
    "UPDAT_DT",
    "ANLNG_S_TP_HSD_TP_STRIP"
]

if all(col in filtered.columns for col in required_cols):

    special_df = filtered[required_cols].copy()

    # Konversi data
    special_df["UPDAT_DT"] = pd.to_datetime(
        special_df["UPDAT_DT"],
        errors="coerce"
    )

    special_df["ANLNG_S_TP_HSD_TP_STRIP"] = pd.to_numeric(
        special_df["ANLNG_S_TP_HSD_TP_STRIP"],
        errors="coerce"
    )

    # Hapus data kosong
    special_df = special_df.dropna(
        subset=required_cols
    )

    # Urutkan berdasarkan waktu
    special_df = special_df.sort_values(
        "UPDAT_DT"
    ).reset_index(drop=True)

    if not special_df.empty:

        # =========================================
        # PENGATURAN BATAS DINAMIS
        # =========================================

        st.markdown("### Pengaturan Batas Monitoring")

        c1, c2 = st.columns(2)

        upper_special = c1.number_input(
            "Upper Limit",
            value=820.0,
            step=1.0,
            format="%.3f",
            key="special_upper_limit"
        )

        lower_special = c2.number_input(
            "Lower Limit",
            value=780.0,
            step=1.0,
            format="%.3f",
            key="special_lower_limit"
        )

        if upper_special <= lower_special:

            st.error(
                "Upper Limit harus lebih besar dari Lower Limit."
            )

        else:

            # =====================================
            # STATUS DATA
            # =====================================

            value_col = "ANLNG_S_TP_HSD_TP_STRIP"

            special_df["STATUS"] = np.select(
                [
                    special_df[value_col] > upper_special,
                    special_df[value_col] < lower_special
                ],
                [
                    "Di atas batas atas",
                    "Di bawah batas bawah"
                ],
                default="Normal"
            )

            # =====================================
            # GRAFIK
            # =====================================

            fig_special = go.Figure()

            # Garis penghubung setiap coil
            fig_special.add_trace(
                go.Scatter(
                    x=special_df["UPDAT_DT"],
                    y=special_df[value_col],
                    mode="lines",
                    name="Trend",
                    line=dict(
                        color="blue",
                        width=1.5
                    ),
                    hoverinfo="skip"
                )
            )

            # Titik data berdasarkan status
            color_map = {
                "Normal": "green",
                "Di atas batas atas": "red",
                "Di bawah batas bawah": "red"
            }

            for status, color in color_map.items():

                status_df = special_df[
                    special_df["STATUS"] == status
                ]

                if not status_df.empty:

                    fig_special.add_trace(
                        go.Scatter(
                            x=status_df["UPDAT_DT"],
                            y=status_df[value_col],
                            mode="markers",
                            name=status,
                            marker=dict(
                                color=color,
                                size=7
                            ),
                            customdata=status_df[
                                ["STATUS"]
                            ].to_numpy(),
                            hovertemplate=(
                                "UPDAT_DT: %{x|%d-%m-%Y %H:%M:%S}<br>"
                                "Nilai: %{y:.3f}<br>"
                                "Status: %{customdata[0]}"
                                "<extra></extra>"
                            )
                        )
                    )

            # =====================================
            # GARIS BATAS
            # =====================================

            fig_special.add_hline(
                y=upper_special,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"USL = {upper_special:g}",
                annotation_position="top right"
            )

            fig_special.add_hline(
                y=lower_special,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"LSL = {lower_special:g}",
                annotation_position="bottom right"
            )

            # =====================================
            # LAYOUT
            # =====================================

            fig_special.update_layout(
                height=600,
                hovermode="closest",
                title="Trend ANLNG_S_TP_HSD_TP_STRIP",
                xaxis_title="UPDAT_DT (Tanggal dan Waktu)",
                yaxis_title=value_col,
                legend_title="Status",
                margin=dict(
                    l=50,
                    r=40,
                    t=70,
                    b=80
                )
            )

            fig_special.update_xaxes(
                type="date",
                tickformat="%d-%m-%Y<br>%H:%M:%S",
                showgrid=True,
                rangeslider_visible=True
            )

            fig_special.update_yaxes(
                showgrid=True
            )

            # =====================================
            # TAMPILKAN GRAFIK
            # =====================================

            st.plotly_chart(
                fig_special,
                use_container_width=True,
                key="special_strip_chart"
            )

            # =====================================
            # RINGKASAN MONITORING
            # =====================================

            total_data = len(special_df)

            above_limit = (
                special_df[value_col] > upper_special
            ).sum()

            below_limit = (
                special_df[value_col] < lower_special
            ).sum()

            normal_data = (
                total_data - above_limit - below_limit
            )

            st.markdown("### Ringkasan Monitoring")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Total Data",
                f"{total_data:,}"
            )

            c2.metric(
                "Normal",
                f"{normal_data:,}"
            )

            c3.metric(
                "Di atas Upper Limit",
                f"{above_limit:,}"
            )

            c4.metric(
                "Di bawah Lower Limit",
                f"{below_limit:,}"
            )

    else:

        st.warning(
            "Tidak ada data valid untuk grafik monitoring."
        )

else:

    st.warning(
        "Kolom UPDAT_DT atau ANLNG_S_TP_HSD_TP_STRIP "
        "tidak ditemukan."
    )


# =====================================================
# TAB 3: QUALITY DAN DEFECT
# =====================================================
with tab_quality:
    st.subheader("Distribusi Grade dan Defect")

    quality_cols = [
        c for c in [
            "TOP_SRFCE_GRADE_CD",
            "SHAPE_GRADE_CD",
            "BOTTM_SRFCE_GRADE_CD",
            "NO1_DEFCT_CD",
            "NO2_DEFCT_CD",
            "NO3_DEFCT_CD",
            "RSLT_CD",
            "NO1_TRG_GRADE_CD",
        ]
        if c in filtered.columns
    ]

    if quality_cols:
        quality_col = st.selectbox(
            "Pilih kategori kualitas/defect",
            quality_cols,
        )

        quality_df = filtered.copy()
        quality_df[quality_col] = (
            quality_df[quality_col]
            .fillna("(Kosong)")
            .astype(str)
        )

        counts = (
            quality_df.groupby(quality_col, dropna=False)
            .size()
            .reset_index(name="Jumlah")
            .sort_values("Jumlah", ascending=False)
        )

        fig = px.bar(
            counts,
            x=quality_col,
            y="Jumlah",
            color=quality_col,
            text="Jumlah",
            title=f"Distribusi {quality_col}",
            hover_data=["Jumlah"],
        )

        fig.update_layout(
            height=450,
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="quality_chart",
        )

        st.dataframe(
            counts,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Kolom grade atau defect tidak ditemukan.")

    # Scatter hubungan ketebalan dan berat
    if "INLET_COIL_TH" in filtered.columns and "COIL_WT" in filtered.columns:
        st.subheader("Hubungan Ketebalan dan Berat Coil")

        scatter_color = (
            "NO1_TRG_GRADE_CD"
            if "NO1_TRG_GRADE_CD" in filtered.columns
            else None
        )

        fig = px.scatter(
            filtered,
            x="INLET_COIL_TH",
            y="COIL_WT",
            color=scatter_color,
            hover_data=hover_columns,
            title="Ketebalan vs Coil Weight",
        )

        fig.update_layout(height=500)

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="thickness_weight_chart",
        )


# =====================================================
# TAB 4: TABEL DATA DAN EKSPOR
# =====================================================
with tab_data:
    st.subheader("Data sesuai filter")

    st.write(
        f"Menampilkan {len(filtered):,} record."
    )

    display_df = filtered.drop(
        columns=["_ANALYSIS_DATE"],
        errors="ignore",
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=500,
        hide_index=True,
    )

    csv_bytes = display_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="coil_data_filtered.csv",
        mime="text/csv",
    )

    excel_buffer = BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl",
    ) as writer:
        display_df.to_excel(
            writer,
            index=False,
            sheet_name="Filtered Data",
        )

    st.download_button(
        "⬇️ Download Excel",
        data=excel_buffer.getvalue(),
        file_name="coil_data_filtered.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )
