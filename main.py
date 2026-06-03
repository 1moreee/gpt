import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# ─────────────────────────────────────────────
#  НАЛАШТУВАННЯ СТОРІНКИ
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CyberTracker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEVERITY_ORDER  = ["low", "medium", "high", "critical"]
SEVERITY_COLORS = {
    "low":      "#3fb950",
    "medium":   "#e3b341",
    "high":     "#f78166",
    "critical": "#ff4444",
}
COLS = ["date", "type", "severity", "description"]

# ─────────────────────────────────────────────
#  SESSION STATE — єдине надійне сховище
#  Streamlit перезапускає скрипт при КОЖНІЙ дії,
#  тому зберігаємо DataFrame прямо у session_state.
#  Він живе весь час поки відкрита вкладка браузера.
# ─────────────────────────────────────────────
if "incidents" not in st.session_state:
    st.session_state["incidents"] = pd.DataFrame(columns=COLS)


def get_df() -> pd.DataFrame:
    return st.session_state["incidents"]


def add_incident(date_str: str, type_inc: str, severity: str, desc: str) -> None:
    new_row = pd.DataFrame([{
        "date":        date_str,
        "type":        type_inc,
        "severity":    severity,
        "description": desc,
    }])
    st.session_state["incidents"] = pd.concat(
        [st.session_state["incidents"], new_row],
        ignore_index=True,
    )


def export_csv() -> bytes:
    """Генерує CSV для завантаження."""
    return get_df().to_csv(index=False).encode("utf-8")


def import_csv(uploaded) -> int:
    """Завантажує інциденти з CSV-файлу. Повертає кількість доданих рядків."""
    try:
        df_new = pd.read_csv(uploaded, dtype=str)
        for col in COLS:
            if col not in df_new.columns:
                return 0
        df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce")
        df_new = df_new.dropna(subset=["date"])
        df_new["date"] = df_new["date"].dt.strftime("%Y-%m-%d")
        st.session_state["incidents"] = pd.concat(
            [st.session_state["incidents"], df_new[COLS]],
            ignore_index=True,
        )
        return len(df_new)
    except Exception:
        return 0


# ─────────────────────────────────────────────
#  СТИЛІ
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

.stApp {
    background-color: #060a10;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(88,166,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(188,140,255,0.06) 0%, transparent 55%);
    color: #cdd9e5;
    font-family: 'Syne', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }

.cyber-header {
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
    margin-top: 4px;
}
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
[data-testid="stForm"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 28px !important;
}
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
[data-testid="stDataFrame"] { border: 1px solid #21262d !important; border-radius: 10px !important; }
[data-testid="stExpander"]  { background: #0d1117 !important; border: 1px solid #21262d !important; border-radius: 10px !important; }
hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)


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

    df_all = get_df()
    total    = len(df_all)
    high_cnt = len(df_all[df_all["severity"].isin(["high","critical"])]) if total else 0

    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;
                color:#484f58;letter-spacing:1px;line-height:2.2;'>
        ВСЬОГО ІНЦИДЕНТІВ<br>
        <span style='font-size:1.6rem;color:#58a6ff;font-weight:700;'>{total}</span>
        <br><br>КРИТИЧНИХ / ВИСОКИХ<br>
        <span style='font-size:1.3rem;color:#f78166;font-weight:700;'>{high_cnt}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Імпорт CSV ──
    st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                "letter-spacing:2px;color:#484f58;'>ІМПОРТ / ЕКСПОРТ</div>",
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Завантажити CSV", type="csv", label_visibility="collapsed")
    if uploaded:
        n = import_csv(uploaded)
        if n:
            st.success(f"Додано {n} записів")
        else:
            st.error("Невірний формат CSV")

    if total > 0:
        st.download_button(
            "⬇ Зберегти CSV",
            data=export_csv(),
            file_name="cyber_incidents.csv",
            mime="text/csv",
            use_container_width=True,
        )


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
            date     = st.date_input("📅 Дата виявлення", value=datetime.today())
            type_inc = st.selectbox("⚡ Тип загрози", [
                "Phishing", "Malware", "DDoS", "SQL Injection",
                "Ransomware", "Brute Force", "Zero-Day", "Insider Threat",
            ])
        with col2:
            severity = st.select_slider(
                "🔥 Рівень критичності",
                options=SEVERITY_ORDER,
                value="medium",
            )
            sev_color = SEVERITY_COLORS.get(severity, "#58a6ff")
            st.markdown(f"""
            <div style='margin-top:8px;padding:10px 16px;
                        background:{sev_color}18;border:1px solid {sev_color}44;
                        border-radius:8px;font-family:JetBrains Mono,monospace;
                        font-size:0.85rem;color:{sev_color};letter-spacing:1px;'>
                ● THREAT LEVEL: {severity.upper()}
            </div>
            """, unsafe_allow_html=True)

        desc      = st.text_area("📝 Опис інциденту",
                                 placeholder="Обставини виявлення, вплив, вжиті заходи...",
                                 height=120)
        submitted = st.form_submit_button("⬆ ЗАРЕЄСТРУВАТИ ІНЦИДЕНТ", use_container_width=True)

    if submitted:
        add_incident(date.strftime("%Y-%m-%d"), type_inc, severity, desc)
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

    # Останні записи
    df_recent = get_df()
    if not df_recent.empty:
        st.markdown("<div class='section-label'>// ОСТАННІ ЗАПИСИ</div>",
                    unsafe_allow_html=True)
        for _, row in df_recent.iloc[::-1].head(5).iterrows():
            sev       = row["severity"]
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

    df = get_df().copy()

    if df.empty:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;'>
            <div style='font-family:JetBrains Mono,monospace;font-size:3rem;color:#21262d;margin-bottom:16px;'>◌</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:0.85rem;color:#484f58;letter-spacing:2px;'>
                БАЗА ДАНИХ ПОРОЖНЯ.<br>ЗАРЕЄСТРУЙТЕ ПЕРШИЙ ІНЦИДЕНТ.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # ── МЕТРИКИ ────────────────────────────────
    total     = len(df)
    critical  = len(df[df["severity"] == "critical"])
    high      = len(df[df["severity"] == "high"])
    most_type = df["type"].value_counts().idxmax()
    last_date = df["date"].max().strftime("%d.%m.%Y")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Всього",      total)
    c2.metric("Critical",    critical)
    c3.metric("High",        high)
    c4.metric("Топ загроза", most_type)
    c5.metric("Останній",    last_date)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ЧАСОВИЙ ГРАФІК ─────────────────────────
    st.markdown("<div class='section-label'>// ДИНАМІКА ІНЦИДЕНТІВ У ЧАСІ</div>",
                unsafe_allow_html=True)

    timeline = (
        df.groupby(df["date"].dt.date).size()
        .reset_index(name="Кількість")
        .rename(columns={"date": "Дата"})
        .sort_values("Дата")
    )
    fig_line = px.area(timeline, x="Дата", y="Кількість",
                       color_discrete_sequence=["#58a6ff"], template="plotly_dark")
    fig_line.update_traces(fill="tozeroy", fillcolor="rgba(88,166,255,0.10)",
                           line=dict(width=2, color="#58a6ff"))
    fig_line.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono", color="#484f58", size=11),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor="#21262d", linecolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#21262d", tickformat="d"),
        hovermode="x unified", height=240,
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ── ТИПИ + КРИТИЧНІСТЬ ─────────────────────
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("<div class='section-label'>// ТИПИ ЗАГРОЗ</div>",
                    unsafe_allow_html=True)
        tc = df["type"].value_counts().reset_index()
        tc.columns = ["Тип", "Кількість"]
        fig_bar = px.bar(tc, x="Кількість", y="Тип", orientation="h",
                         color="Кількість",
                         color_continuous_scale=[[0,"#1f2937"],[1,"#58a6ff"]],
                         template="plotly_dark", text="Кількість")
        fig_bar.update_traces(textfont=dict(family="JetBrains Mono", size=11,
                                            color="#cdd9e5"), textposition="outside")
        fig_bar.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font=dict(family="JetBrains Mono", color="#484f58", size=11),
            margin=dict(l=0, r=20, t=10, b=0),
            xaxis=dict(gridcolor="#21262d"), coloraxis_showscale=False,
            yaxis=dict(gridcolor="rgba(0,0,0,0)", categoryorder="total ascending"),
            height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.markdown("<div class='section-label'>// КРИТИЧНІСТЬ</div>",
                    unsafe_allow_html=True)
        sc = df["severity"].value_counts().reset_index()
        sc.columns = ["Рівень", "Кількість"]
        sc["_o"] = sc["Рівень"].map({s: i for i, s in enumerate(SEVERITY_ORDER)})
        sc = sc.sort_values("_o").drop(columns="_o")
        colors = [SEVERITY_COLORS.get(s, "#58a6ff") for s in sc["Рівень"]]

        fig_donut = go.Figure(go.Pie(
            labels=sc["Рівень"].str.upper(), values=sc["Кількість"], hole=0.55,
            marker=dict(colors=colors, line=dict(color="#0d1117", width=3)),
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig_donut.add_annotation(text=f"<b>{total}</b>", x=0.5, y=0.5,
                                  showarrow=False,
                                  font=dict(size=28, color="#58a6ff",
                                            family="JetBrains Mono"))
        fig_donut.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font=dict(family="JetBrains Mono", color="#484f58", size=11),
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(bgcolor="#0d1117", bordercolor="#21262d",
                        font=dict(family="JetBrains Mono", size=11)),
            height=300,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── МАТРИЦЯ ЗАГРОЗ ─────────────────────────
    st.markdown("<div class='section-label'>// МАТРИЦЯ ЗАГРОЗ (ТИП × КРИТИЧНІСТЬ)</div>",
                unsafe_allow_html=True)
    pivot = (
        df.groupby(["type","severity"]).size().reset_index(name="n")
        .pivot(index="type", columns="severity", values="n")
        .reindex(columns=[c for c in SEVERITY_ORDER if c in df["severity"].unique()])
        .fillna(0).astype(int)
    )
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[c.upper() for c in pivot.columns],
        y=pivot.index,
        colorscale=[[0,"#0d1117"],[0.3,"#1f3a5f"],[0.7,"#1f6feb"],[1,"#58a6ff"]],
        text=pivot.values, texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono", size=13, color="#cdd9e5"),
        showscale=False,
    ))
    fig_heat.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono", color="#484f58", size=11),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(side="top", linecolor="#21262d"),
        yaxis=dict(linecolor="#21262d", autorange="reversed"),
        height=max(200, len(pivot)*48+60),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── ТАБЛИЦЯ ────────────────────────────────
    with st.expander("📋 Всі зареєстровані інциденти"):
        disp = df.copy()
        disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
        disp = disp.sort_values("date", ascending=False)
        disp.columns = ["Дата","Тип загрози","Критичність","Опис"]
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
                color:#30363d;letter-spacing:2px;text-align:center;padding:8px 0;'>
        CYBERTRACKER · КІБЕРБЕЗПЕКА ТА ЗАХИСТ ІНФОРМАЦІЇ
    </div>
    """, unsafe_allow_html=True)
