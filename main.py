import streamlit as st
import pandas as pd
import os

# Налаштування сторінки
st.set_page_config(page_title="Cyber Tracker", layout="wide")

DATA_FILE = "data.csv"

# Функція завантаження даних
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            # errors='coerce' перетворить нечитабельні дати в NaT (порожньо), замість помилки
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Видаляємо рядки, де дата не змогла розпізнатися
            df = df.dropna(subset=['date'])
            return df
        except Exception as e:
            st.error(f"Помилка читання файлу: {e}")
            return pd.DataFrame(columns=["date", "type", "severity", "description"])
    return pd.DataFrame(columns=["date", "type", "severity", "description"])

# --- БОКОВА ПАНЕЛЬ ---
st.sidebar.header("Меню керування")
page = st.sidebar.selectbox("Перейти до розділу:", ["Реєстрація інцидентів", "Аналітика та порівняння"])

# --- СТОРІНКА 1: ДОДАТИ ІНЦИДЕНТ ---
if page == "Реєстрація інцидентів":
    st.title("Новий запис про кіберінцидент")
    
    with st.form("incident_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("Дата виявлення")
            type_inc = st.selectbox("Тип загрози", ["Phishing", "Malware", "DDoS", "SQL Injection", "Ransomware"])
        with col2:
            severity = st.select_slider("Рівень важливості", options=["low", "medium", "high"])
        
        desc = st.text_area("Детальний опис інциденту")
        submit = st.form_submit_button("Зберегти дані")

    if submit:
        # Зберігаємо дату у стандартному форматі ISO (YYYY-MM-DD)
        new_row = {
            "date": date.strftime('%Y-%m-%d'), 
            "type": type_inc,
            "severity": severity,
            "description": desc
        }
        df_new = pd.DataFrame([new_row])
        df_existing = load_data()
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
        df_final.to_csv(DATA_FILE, index=False)
        st.success("Інцидент успішно зареєстровано!")

# --- СТОРІНКА 2: АНАЛІТИКА ТА ПОРІВНЯННЯ ---
elif page == "Аналітика та порівняння":
    st.title("Аналітичний дашборд")
    
    df = load_data()

    if not df.empty:
        st.metric("Загальна кількість зафіксованих подій", len(df))
        st.divider()

        # ЛІНІЙНИЙ ГРАФІК
        st.subheader("Динаміка інцидентів у часі")
        # Групуємо та сортуємо за датою
        timeline_df = df.groupby('date').size().reset_index(name='Кількість')
        timeline_df = timeline_df.sort_values('date').set_index('date')
        
        if not timeline_df.empty:
            st.line_chart(timeline_df)
        else:
            st.info("Недостатньо коректних дат для побудови графіка.")

        # СЕКЦІЯ ПОРІВНЯННЯ
        st.subheader("Порівняння параметрів")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("Кількість за типами:")
            st.bar_chart(df["type"].value_counts())
            
        with col_right:
            st.write("Розподіл за критичністю:")
            st.table(df.groupby('severity').size().reset_index(name='Кількість'))

        with st.expander("Переглянути всі дані"):
            st.dataframe(df, use_container_width=True)
            
    else:
        st.warning("База даних порожня або містить некоректні дані.")
