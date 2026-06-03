import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# ─────────────────────────────────────────────
#  НАЛАШТУВАННЯ СТОРІНКИ
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CyberTracker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = "data.csv"

SEVERITY_ORDER = ["low", "medium", "high", "critical"]
SEVERITY_COLORS = {
    "low":      "#3fb950",
    "medium":   "#e3b341",
    "high":     "#f78166",
    "critical": "#ff4444",
}
TYPE_COLORS = ["#58a6ff", "#bc8cff", "#3fb950", "#e3b341", "#f78166", "#79c0ff"]

# ─────────────────────────────────────────────
#  СТИЛІ
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── ЗАГАЛЬНИЙ ФОН ── */
.stApp {
    background-color: #060a10;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(88,166,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(188,140,255,0.06) 0%, transparent 55%);
    color: #cdd9e5;
    font-family: 'Syne', sans-serif;
}

/* ── БІЧНА ПАНЕЛЬ ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }

/* ── ЗАГОЛОВОК ДАШБОРДУ ── */
.cyber-header {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 28px 0 20px 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 28px;
}
.cyber-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: #58a6ff;
    text-shadow: 0 0 24px rgba(88,166,255,0.35);
}
.cyber-sub {
    font-size: 0.82rem;
    color: #484f58;
    letter-spacing: 3px;
    text-transform: uppercase;
}

/* ── МЕТРИКИ ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 20px 24px !important;
    transition: border-color .2s;
}
[data-testid="stMetric"]:hover { border-color: #58a6ff55; }
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #58a6ff !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #484f58 !important;
}

/* ── ФОРМА ── */
[data-testid="stForm"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 28px !important;
}

/* ── КНОПКА ── */
.stButton > button {
    background: linear-gradient(90deg, #1f6feb 0%, #388bfd 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    padding: 10px 32px !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── СЕКЦІЙНИЙ ЗАГОЛОВОК ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #484f58;
    padding: 18px 0 8px 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 16px;
}

/* ── БЕЙДЖИ SEVERITY ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-low      { background:#3fb95022; color:#3fb950; border:1px solid #3fb95055; }
.badge-medium   { background:#e3b34122; color:#e3b341; border:1px solid #e3b34155; }
.badge-high     { background:#f7816622; color:#f78166; border:1px solid #f7816655; }
.badge-critical { background:#ff444422; color:#ff4444; border:1px solid #ff444455; }

/* ── ТАБЛИЦЯ ── */
[data-testid="stDataFrame"] {
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
}

/* ── DIVIDER ── */
hr { border-color: #21262d !important; }

/* ── INPUT/SELECT ── */
.stSelectbox > div > div,
.stTextArea > div > div,
.stDateInput > div > div {
    background: #161b22 !important;
    border-color: #30363d !important;
    border-radius: 8px !important;
    color: #cdd9e5 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ДАНІ
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Читає CSV і повертає DataFrame.
    Виправлення: явно задаємо dtype для колонок,
    щоб при concat не губився формат дати.
    """
    cols = ["date", "type", "severity", "description"]
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE, dtype=str)   # читаємо ВСЕ як рядки
            # Приводимо до потрібних типів після читання
            for col in cols:
                if col not in df.columns:
                    df[col] = ""
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            return df[cols]
        except Exception as e:
            st.error(f"Помилка читання файлу: {e}")
    return pd.DataFrame(columns=cols)


def save_incident(date_str: str, type_inc: str, severity: str, desc: str) -> None:
    """
    Додає один рядок до CSV.
    Виправлення: замість read→concat→write використовуємо
    дозапис одного рядка (режим 'a'), що унеможливлює
    втрату попередніх записів через помилку типів.
    """
    file_exists = os.path.exists(DATA_FILE)
    new_row = pd.DataFrame([{
        "date":        date_str,
        "type":        type_inc,
        "severity":    severity,
        "description": desc,
    }])
    new_row.to_csv(
        DATA_FILE,
        mode="a",               # дозапис, а не перезапис
        header=not file_exists, # заголовок лише при першому записі
        index=False,
    )


# ─────────────────────────────────────────────
#  БІЧНА ПАНЕЛЬ
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:18px 0 10px 0;'>
        <span style='font-family:JetBrains Mono,monospace;font-size:1.15rem;
                     font-weight:700;color:#58a6ff;letter-spacing:2px;'>
            🛡️ CYBER<br>TRACKER
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Навігація",
        ["📋 Реєстрація", "📊 Аналітика"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    df_sidebar = load_data()
    total = len(df_sidebar)
    high_cnt = len(df_sidebar[df_sidebar["severity"].isin(["high", "critical"])]) if total else 0

    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;
                color:#484f58;letter-spacing:1px;line-height:2.2;'>
        ВСЬОГО ІНЦИДЕНТІВ<br>
        <span style='font-size:1.6rem;color:#58a6ff;font-weight:700;'>{total}</span>
        <br><br>
        КРИТИЧНИХ / ВИСОКИХ<br>
        <span style='font-size:1.3rem;color:#f78166;font-weight:700;'>{high_cnt}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  СТОРІНКА 1 — РЕЄСТРАЦІЯ
# ─────────────────────────────────────────────
if page == "📋 Реєстрація":
    st.markdown("""
    <div class='cyber-header'>
        <div class='cyber-title'>// РЕЄСТРАЦІЯ ІНЦИДЕНТУ</div>
        <div class='cyber-sub'>Cyber Threat Registration System</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("incident_form", clear_on_submit=True):
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            date = st.date_input("📅 Дата виявлення", value=datetime.today())
            type_inc = st.selectbox(
                "⚡ Тип загрози",
                ["Phishing", "Malware", "DDoS", "SQL Injection",
                 "Ransomware", "Brute Force", "Zero-Day", "Insider Threat"],
            )

        with col2:
            severity = st.select_slider(
                "🔥 Рівень критичності",
                options=SEVERITY_ORDER,
                value="medium",
            )
            # Кольоровий індикатор рівня
            sev_color = SEVERITY_COLORS.get(severity, "#58a6ff")
            st.markdown(f"""
            <div style='margin-top:8px;padding:10px 16px;
                        background:{sev_color}18;border:1px solid {sev_color}44;
                        border-radius:8px;font-family:JetBrains Mono,monospace;
                        font-size:0.85rem;color:{sev_color};letter-spacing:1px;'>
                ● THREAT LEVEL: {severity.upper()}
            </div>
            """, unsafe_allow_html=True)

        desc = st.text_area(
            "📝 Детальний опис інциденту",
            placeholder="Опишіть обставини виявлення, потенційний вплив та вжиті заходи...",
            height=130,
        )

        submitted = st.form_submit_button("⬆ ЗАРЕЄСТРУВАТИ ІНЦИДЕНТ", use_container_width=True)

    if submitted:
        save_incident(
            date_str  = date.strftime("%Y-%m-%d"),
            type_inc  = type_inc,
            severity  = severity,
            desc      = desc,
        )
        sev_color = SEVERITY_COLORS.get(severity, "#58a6ff")
        st.markdown(f"""
        <div style='margin-top:16px;padding:16px 20px;
                    background:{sev_color}12;border:1px solid {sev_color}44;
                    border-radius:10px;font-family:JetBrains Mono,monospace;'>
            <span style='color:{sev_color};font-size:1rem;'>✓ ІНЦИДЕНТ ЗАРЕЄСТРОВАНО</span><br>
            <span style='color:#484f58;font-size:0.78rem;'>
                {date.strftime("%Y-%m-%d")} &nbsp;|&nbsp; {type_inc} &nbsp;|&nbsp; {severity.upper()}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Останні 5 інцидентів
    df_recent = load_data()
    if not df_recent.empty:
        st.markdown("<div class='section-label'>// ОСТАННІ ЗАПИСИ</div>", unsafe_allow_html=True)
        recent = df_recent.sort_values("date", ascending=False).head(5)
        for _, row in recent.iterrows():
            sev = row["severity"]
            sev_color = SEVERITY_COLORS.get(sev, "#58a6ff")
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:16px;
                        padding:12px 16px;margin-bottom:8px;
                        background:#0d1117;border:1px solid #21262d;
                        border-left:3px solid {sev_color};border-radius:8px;'>
                <span style='font-family:JetBrains Mono,monospace;font-size:0.78rem;
                             color:#484f58;min-width:90px;'>{str(row['date'])[:10]}</span>
                <span style='font-family:JetBrains Mono,monospace;font-size:0.85rem;
                             color:#cdd9e5;flex:1;'>{row['type']}</span>
                <span class='badge badge-{sev}'>{sev}</span>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  СТОРІНКА 2 — АНАЛІТИКА
# ─────────────────────────────────────────────
elif page == "📊 Аналітика":
    st.markdown("""
    <div class='cyber-header'>
        <div class='cyber-title'>// АНАЛІТИЧНИЙ ДАШБОРД</div>
        <div class='cyber-sub'>Threat Intelligence Overview</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()

    if df.empty:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;'>
            <div style='font-family:JetBrains Mono,monospace;font-size:3rem;
                        color:#21262d;margin-bottom:16px;'>◌</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:0.85rem;
                        color:#484f58;letter-spacing:2px;'>
                БАЗА ДАНИХ ПОРОЖНЯ.<br>ЗАРЕЄСТРУЙТЕ ПЕРШИЙ ІНЦИДЕНТ.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── МЕТРИКИ ────────────────────────────────
    total      = len(df)
    critical   = len(df[df["severity"] == "critical"])
    high       = len(df[df["severity"] == "high"])
    most_type  = df["type"].value_counts().idxmax() if total else "—"
    last_date  = df["date"].max().strftime("%d.%m.%Y") if total else "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Всього інцидентів", total)
    c2.metric("Critical",  critical,  delta=None)
    c3.metric("High",      high,      delta=None)
    c4.metric("Топ загроза", most_type)
    c5.metric("Останній",  last_date)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ЧАСОВИЙ ГРАФІК ─────────────────────────
    st.markdown("<div class='section-label'>// ДИНАМІКА ІНЦИДЕНТІВ У ЧАСІ</div>",
                unsafe_allow_html=True)

    timeline = (
        df.groupby(df["date"].dt.date)
        .size()
        .reset_index(name="Кількість")
        .rename(columns={"date": "Дата"})
        .sort_values("Дата")
    )

    fig_line = px.area(
        timeline, x="Дата", y="Кількість",
        color_discrete_sequence=["#58a6ff"],
        template="plotly_dark",
    )
    fig_line.update_traces(
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.10)",
        line=dict(width=2, color="#58a6ff"),
    )
    fig_line.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono", color="#484f58", size=11),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor="#21262d", linecolor="#21262d", showgrid=True),
        yaxis=dict(gridcolor="#21262d", linecolor="#21262d", showgrid=True,
                   tickformat="d"),
        hovermode="x unified",
        height=260,
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ── ДВА ГРАФІКИ В РЯД ──────────────────────
    col_l, col_r = st.columns(2, gap="large")

    # Типи загроз — горизонтальна барна діаграма
    with col_l:
        st.markdown("<div class='section-label'>// РОЗПОДІЛ ЗА ТИПАМИ ЗАГРОЗ</div>",
                    unsafe_allow_html=True)
        type_counts = df["type"].value_counts().reset_index()
        type_counts.columns = ["Тип", "Кількість"]

        fig_bar = px.bar(
            type_counts, x="Кількість", y="Тип",
            orientation="h",
            color="Кількість",
            color_continuous_scale=[[0, "#1f2937"], [1, "#58a6ff"]],
            template="plotly_dark",
            text="Кількість",
        )
        fig_bar.update_traces(
            textfont=dict(family="JetBrains Mono", size=11, color="#cdd9e5"),
            textposition="outside",
        )
        fig_bar.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font=dict(family="JetBrains Mono", color="#484f58", size=11),
            margin=dict(l=0, r=20, t=10, b=0),
            xaxis=dict(gridcolor="#21262d", linecolor="#21262d"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor="#21262d",
                       categoryorder="total ascending"),
            coloraxis_showscale=False,
            height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Рівні критичності — donut
    with col_r:
        st.markdown("<div class='section-label'>// РОЗПОДІЛ ЗА КРИТИЧНІСТЮ</div>",
                    unsafe_allow_html=True)
        sev_counts = df["severity"].value_counts().reset_index()
        sev_counts.columns = ["Рівень", "Кількість"]
        # Сортуємо за SEVERITY_ORDER
        sev_counts["_order"] = sev_counts["Рівень"].map(
            {s: i for i, s in enumerate(SEVERITY_ORDER)}
        )
        sev_counts = sev_counts.sort_values("_order").drop(columns="_order")

        colors = [SEVERITY_COLORS.get(s, "#58a6ff") for s in sev_counts["Рівень"]]

        fig_donut = go.Figure(go.Pie(
            labels=sev_counts["Рівень"].str.upper(),
            values=sev_counts["Кількість"],
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#0d1117", width=3)),
            textfont=dict(family="JetBrains Mono", size=11),
            hovertemplate="<b>%{label}</b><br>%{value} інцидентів<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{total}</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=28, color="#58a6ff", family="JetBrains Mono"),
        )
        fig_donut.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font=dict(family="JetBrains Mono", color="#484f58", size=11),
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(bgcolor="#0d1117", bordercolor="#21262d",
                        font=dict(family="JetBrains Mono", size=11)),
            height=300,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── ТЕПЛОВА КАРТА (тип × критичність) ──────
    st.markdown("<div class='section-label'>// МАТРИЦЯ ЗАГРОЗ</div>",
                unsafe_allow_html=True)

    pivot = (
        df.groupby(["type", "severity"])
        .size()
        .reset_index(name="n")
        .pivot(index="type", columns="severity", values="n")
        .reindex(columns=[c for c in SEVERITY_ORDER if c in df["severity"].unique()])
        .fillna(0)
        .astype(int)
    )

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[c.upper() for c in pivot.columns],
        y=pivot.index,
        colorscale=[[0, "#0d1117"], [0.3, "#1f3a5f"], [0.7, "#1f6feb"], [1, "#58a6ff"]],
        text=pivot.values,
        texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono", size=13, color="#cdd9e5"),
        hovertemplate="<b>%{y}</b> / %{x}<br>%{z} інцидентів<extra></extra>",
        showscale=False,
    ))
    fig_heat.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono", color="#484f58", size=11),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(side="top", linecolor="#21262d"),
        yaxis=dict(linecolor="#21262d", autorange="reversed"),
        height=max(200, len(pivot) * 48 + 60),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── ПОВНА ТАБЛИЦЯ ──────────────────────────
    with st.expander("📋 Всі зареєстровані інциденти"):
        display_df = df.copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df = display_df.sort_values("date", ascending=False)
        display_df.columns = ["Дата", "Тип загрози", "Критичність", "Опис"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── ПІДВАЛ ──────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
                color:#30363d;letter-spacing:2px;text-align:center;padding:8px 0;'>
        CYBERTRACKER · КІБЕРБЕЗПЕКА ТА ЗАХИСТ ІНФОРМАЦІЇ
    </div>
    """, unsafe_allow_html=True)
