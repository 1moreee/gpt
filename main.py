import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st


@dataclass
class AnalysisConfig:
    """
    Клас-конфігурація для зберігання порогових значень,
    які використовуються для виявлення аномалій.
    """

    packet_threshold: int = 1000
    bytes_threshold: int = 1_000_000


class NetworkAnalyzer:
    """
    Клас NetworkAnalyzer інкапсулює всю бізнес-логіку аналізу мережевого трафіку.

    Основна ідея ООП тут полягає в тому, що об'єкт класу зберігає DataFrame з даними
    та надає набір методів для їх очищення, фільтрації, аналізу і пошуку аномалій.
    """

    # Список обов'язкових колонок, без яких аналіз виконати неможливо.
    REQUIRED_COLUMNS: List[str] = [
        "Source IP",
        "Destination IP",
        "Protocol",
        "Port",
        "Size",
        "Time",
    ]

    def __init__(self, dataframe: pd.DataFrame, config: Optional[AnalysisConfig] = None) -> None:
        """
        Конструктор класу.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Початковий DataFrame, що був зчитаний з CSV-файлу.
        config : Optional[AnalysisConfig]
            Конфігурація з порогами для аномалій.
        """
        self.df: pd.DataFrame = dataframe.copy()
        self.config: AnalysisConfig = config or AnalysisConfig()
        self.filtered_df: pd.DataFrame = pd.DataFrame()

        # Одразу після створення об'єкта виконуємо валідацію та підготовку даних.
        self._validate_columns()
        self._prepare_data()

    def _validate_columns(self) -> None:
        """
        Перевіряє, чи всі обов'язкові колонки присутні у DataFrame.

        Raises
        ------
        KeyError
            Якщо хоча б одна обов'язкова колонка відсутня.
        """
        # За допомогою множини (set) знаходимо відсутні колонки.
        current_columns: Set[str] = set(self.df.columns)
        required_columns: Set[str] = set(self.REQUIRED_COLUMNS)
        missing_columns: Set[str] = required_columns - current_columns

        if missing_columns:
            raise KeyError(
                f"У CSV-файлі відсутні обов'язкові колонки: {', '.join(sorted(missing_columns))}"
            )

    def _prepare_data(self) -> None:
        """
        Проводить попередню підготовку даних:
        - конвертацію часу в datetime;
        - конвертацію Port і Size в числовий тип;
        - нормалізацію назв протоколів;
        - видалення рядків з критично пошкодженими значеннями.
        """
        # Копіюємо DataFrame, щоб уникнути небажаних побічних ефектів.
        self.df = self.df.copy()

        # Перетворюємо колонку часу в тип datetime.
        self.df["Time"] = pd.to_datetime(self.df["Time"], errors="coerce")

        # Перетворюємо розмір пакета та порт у числовий формат.
        # Якщо значення некоректне, pandas підставить NaN.
        self.df["Port"] = pd.to_numeric(self.df["Port"], errors="coerce")
        self.df["Size"] = pd.to_numeric(self.df["Size"], errors="coerce")

        # Нормалізуємо назви протоколів: забираємо пробіли та переводимо у верхній регістр.
        self.df["Protocol"] = self.df["Protocol"].astype(str).str.strip().str.upper()

        # Видаляємо рядки, де відсутні критично важливі дані.
        self.df.dropna(
            subset=["Source IP", "Destination IP", "Protocol", "Size", "Time"],
            inplace=True,
        )

        # Для зручності приводимо колонку порту до цілого типу там, де це можливо.
        self.df["Port"] = self.df["Port"].fillna(0).astype(int)

        # Зберігаємо первинно відфільтрований DataFrame як повну копію очищених даних.
        self.filtered_df = self.df.copy()

    @classmethod
    def from_csv(cls, uploaded_file, config: Optional[AnalysisConfig] = None) -> "NetworkAnalyzer":
        """
        Фабричний метод для створення об'єкта класу з CSV-файлу.

        Parameters
        ----------
        uploaded_file : UploadedFile
            Файл, завантажений через Streamlit.
        config : Optional[AnalysisConfig]
            Додаткова конфігурація.

        Returns
        -------
        NetworkAnalyzer
            Готовий до роботи екземпляр аналізатора.

        Raises
        ------
        pd.errors.EmptyDataError
            Якщо файл порожній.
        UnicodeDecodeError
            Якщо кодування не вдалося розпізнати.
        KeyError
            Якщо відсутні необхідні колонки.
        Exception
            Для інших неочікуваних помилок читання.
        """
        # Зчитуємо байти файлу лише один раз, щоб мати змогу спробувати різні кодування.
        file_bytes: bytes = uploaded_file.getvalue()

        # Список кодувань у вигляді списку демонструє використання базових структур даних Python.
        encodings_to_try: List[str] = ["utf-8", "utf-8-sig", "cp1251", "latin-1"]

        last_exception: Optional[Exception] = None

        for encoding in encodings_to_try:
            try:
                dataframe = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
                return cls(dataframe, config=config)
            except UnicodeDecodeError as exc:
                last_exception = exc
                continue
            except pd.errors.EmptyDataError:
                # Цю помилку передаємо вище, бо файл порожній і пробувати інші кодування немає сенсу.
                raise
            except Exception as exc:
                # Зберігаємо останню помилку для діагностики.
                last_exception = exc

        # Якщо жодне кодування не підійшло, піднімаємо помилку вище.
        raise Exception(f"Не вдалося зчитати CSV-файл. Деталі: {last_exception}")

    def apply_filters(
        self,
        protocols: Optional[List[str]] = None,
        ports: Optional[List[int]] = None,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
        ip_query: str = "",
    ) -> pd.DataFrame:
        """
        Застосовує фільтри до основного набору даних.

        Тут також демонструється використання функціонального програмування:
        - lambda-вираз;
        - функції filter();
        - функції map().
        """
        filtered = self.df.copy()

        # Фільтрація за протоколами, якщо користувач щось вибрав.
        if protocols:
            filtered = filtered[filtered["Protocol"].isin(protocols)]

        # Функціональний стиль: очищаємо список портів через map + filter + lambda.
        # 1) map(int, ...) перетворює значення у числа;
        # 2) filter(lambda ...) залишає лише коректні, невід'ємні порти.
        if ports:
            safe_ports: List[int] = list(
                filter(lambda p: p >= 0, map(int, ports))
            )
            filtered = filtered[filtered["Port"].isin(safe_ports)]

        # Фільтрація за часовим інтервалом.
        if start_time is not None:
            filtered = filtered[filtered["Time"] >= pd.to_datetime(start_time)]
        if end_time is not None:
            filtered = filtered[filtered["Time"] <= pd.to_datetime(end_time)]

        # Пошук за IP-адресою у двох колонках одразу.
        if ip_query.strip():
            ip_query = ip_query.strip()
            filtered = filtered[
                filtered["Source IP"].astype(str).str.contains(ip_query, case=False, na=False)
                | filtered["Destination IP"].astype(str).str.contains(ip_query, case=False, na=False)
            ]

        self.filtered_df = filtered.copy()
        return self.filtered_df

    def get_summary_metrics(self) -> Dict[str, object]:
        """
        Формує основні підсумкові метрики для дашборду.

        Returns
        -------
        Dict[str, object]
            Словник з ключовими показниками.
        """
        df = self.filtered_df

        # Загальний об'єм трафіку у байтах.
        total_traffic: int = int(df["Size"].sum()) if not df.empty else 0

        # Загальна кількість пакетів дорівнює кількості рядків у логах.
        total_packets: int = int(len(df))

        # Унікальні IP-адреси через множини (set).
        unique_source_ips: Set[str] = set(df["Source IP"].astype(str).unique()) if not df.empty else set()
        unique_destination_ips: Set[str] = set(df["Destination IP"].astype(str).unique()) if not df.empty else set()
        all_unique_ips: Set[str] = unique_source_ips.union(unique_destination_ips)

        # Найпоширеніший протокол.
        top_protocol: str = (
            df["Protocol"].mode().iloc[0] if not df.empty and not df["Protocol"].mode().empty else "Н/Д"
        )

        return {
            "total_traffic": total_traffic,
            "total_packets": total_packets,
            "unique_source_ips": len(unique_source_ips),
            "unique_destination_ips": len(unique_destination_ips),
            "all_unique_ips": len(all_unique_ips),
            "top_protocol": top_protocol,
        }

    def get_top_senders(self, top_n: int = 5) -> pd.DataFrame:
        """
        Повертає Топ-N IP-адрес відправників за кількістю пакетів.
        """
        if self.filtered_df.empty:
            return pd.DataFrame(columns=["Source IP", "Packets"])

        result = (
            self.filtered_df.groupby("Source IP")
            .size()
            .reset_index(name="Packets")
            .sort_values(by="Packets", ascending=False)
            .head(top_n)
        )
        return result

    def get_top_receivers(self, top_n: int = 5) -> pd.DataFrame:
        """
        Повертає Топ-N IP-адрес отримувачів за кількістю пакетів.
        """
        if self.filtered_df.empty:
            return pd.DataFrame(columns=["Destination IP", "Packets"])

        result = (
            self.filtered_df.groupby("Destination IP")
            .size()
            .reset_index(name="Packets")
            .sort_values(by="Packets", ascending=False)
            .head(top_n)
        )
        return result

    def get_protocol_distribution(self) -> pd.DataFrame:
        """
        Повертає розподіл трафіку за протоколами.
        Додатково обчислюється як кількість пакетів, так і сумарний обсяг байт.
        """
        if self.filtered_df.empty:
            return pd.DataFrame(columns=["Protocol", "Packets", "Bytes"])

        result = (
            self.filtered_df.groupby("Protocol")
            .agg(Packets=("Protocol", "count"), Bytes=("Size", "sum"))
            .reset_index()
            .sort_values(by="Packets", ascending=False)
        )
        return result

    def get_port_distribution(self, top_n: int = 10) -> pd.DataFrame:
        """
        Повертає Топ-N портів за частотою використання.
        """
        if self.filtered_df.empty:
            return pd.DataFrame(columns=["Port", "Packets"])

        result = (
            self.filtered_df.groupby("Port")
            .size()
            .reset_index(name="Packets")
            .sort_values(by="Packets", ascending=False)
            .head(top_n)
        )
        return result

    def detect_anomalies(self) -> pd.DataFrame:
        """
        Шукає аномальну активність за двома критеріями:
        1. З однієї IP-адреси відправлено надто багато пакетів.
        2. З однієї IP-адреси передано надто великий обсяг байт.

        Returns
        -------
        pd.DataFrame
            Таблиця з підозрілими IP-адресами та поясненням причини.
        """
        if self.filtered_df.empty:
            return pd.DataFrame(columns=["Source IP", "Packets", "Bytes", "Reason"])

        grouped = (
            self.filtered_df.groupby("Source IP")
            .agg(Packets=("Source IP", "count"), Bytes=("Size", "sum"))
            .reset_index()
        )

        # Використовуємо словник для формування текстових причин виявлення аномалій.
        anomaly_reasons: Dict[str, List[str]] = {}

        for _, row in grouped.iterrows():
            reasons: List[str] = []

            if int(row["Packets"]) > self.config.packet_threshold:
                reasons.append(
                    f"перевищено поріг пакетів ({int(row['Packets'])} > {self.config.packet_threshold})"
                )

            if int(row["Bytes"]) > self.config.bytes_threshold:
                reasons.append(
                    f"перевищено поріг байт ({int(row['Bytes'])} > {self.config.bytes_threshold})"
                )

            if reasons:
                anomaly_reasons[str(row["Source IP"])] = reasons

        if not anomaly_reasons:
            return pd.DataFrame(columns=["Source IP", "Packets", "Bytes", "Reason"])

        anomalies = grouped[grouped["Source IP"].astype(str).isin(anomaly_reasons.keys())].copy()

        # До таблиці додаємо текстове пояснення аномалії.
        anomalies["Reason"] = anomalies["Source IP"].astype(str).apply(
            lambda ip: "; ".join(anomaly_reasons.get(ip, []))
        )

        anomalies = anomalies.sort_values(by=["Packets", "Bytes"], ascending=False)
        return anomalies

    def get_filtered_data(self) -> pd.DataFrame:
        """
        Повертає поточний відфільтрований DataFrame.
        """
        return self.filtered_df.copy()

    def get_unique_protocols(self) -> List[str]:
        """
        Повертає список унікальних протоколів для елементів інтерфейсу.
        """
        return sorted(self.df["Protocol"].dropna().astype(str).unique().tolist())

    def get_unique_ports(self) -> List[int]:
        """
        Повертає список унікальних портів для фільтрації.
        """
        return sorted(self.df["Port"].dropna().astype(int).unique().tolist())

    def get_time_range(self) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Повертає мінімальний та максимальний час у наборі даних.
        """
        if self.df.empty:
            return None, None
        return self.df["Time"].min(), self.df["Time"].max()


def format_bytes(size_in_bytes: int) -> str:
    """
    Допоміжна функція для красивого форматування об'єму байт.
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_in_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"


def render_header() -> None:
    """
    Відображає заголовок застосунку та короткий опис.
    """
    st.set_page_config(
        page_title="Аналізатор мережевої активності",
        page_icon="🌐",
        layout="wide",
    )

    st.title("🌐 Формування автоматизованого звіту про мережеву активність")
    st.markdown(
        """
        Цей веб-додаток дозволяє завантажити CSV-файл з мережевими логами,
        відфільтрувати дані за потрібними параметрами та сформувати аналітичний звіт.
        """
    )


def render_sidebar() -> Tuple:
    """
    Формує бічну панель Streamlit та повертає всі вибрані параметри.
    """
    st.sidebar.header("Налаштування аналізу")

    uploaded_file = st.sidebar.file_uploader(
        "Завантажте CSV-файл з логами",
        type=["csv"],
        help="Підтримується лише формат .csv",
    )

    packet_threshold = st.sidebar.number_input(
        "Поріг пакетів для аномалії",
        min_value=1,
        value=1000,
        step=100,
    )

    bytes_threshold = st.sidebar.number_input(
        "Поріг байт для аномалії",
        min_value=1,
        value=1_000_000,
        step=100_000,
    )

    return uploaded_file, int(packet_threshold), int(bytes_threshold)


def render_metrics(metrics: Dict[str, object]) -> None:
    """
    Відображає основні метрики у вигляді карток.
    """
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Загальний трафік", format_bytes(int(metrics["total_traffic"])))
    col2.metric("Кількість пакетів", int(metrics["total_packets"]))
    col3.metric("Унікальні Source IP", int(metrics["unique_source_ips"]))
    col4.metric("Унікальні Destination IP", int(metrics["unique_destination_ips"]))
    col5.metric("Усі унікальні IP", int(metrics["all_unique_ips"]))
    col6.metric("Популярний протокол", str(metrics["top_protocol"]))


def render_protocol_chart(protocol_df: pd.DataFrame) -> None:
    """
    Будує кругову діаграму розподілу трафіку за протоколами.
    """
    if protocol_df.empty:
        st.info("Немає даних для побудови графіка протоколів.")
        return

    fig = px.pie(
        protocol_df,
        names="Protocol",
        values="Packets",
        title="Розподіл трафіку за протоколами",
        hole=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_port_chart(port_df: pd.DataFrame) -> None:
    """
    Будує стовпчикову діаграму для найпопулярніших портів.
    """
    if port_df.empty:
        st.info("Немає даних для побудови графіка портів.")
        return

    fig = px.bar(
        port_df,
        x="Port",
        y="Packets",
        title="Топ портів за кількістю пакетів",
        text="Packets",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_ip_charts(top_senders: pd.DataFrame, top_receivers: pd.DataFrame) -> None:
    """
    Відображає графіки для Топ-5 відправників і отримувачів.
    """
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Топ-5 IP-адрес відправників")
        st.dataframe(top_senders, use_container_width=True)
        if not top_senders.empty:
            fig = px.bar(
                top_senders,
                x="Source IP",
                y="Packets",
                title="Найактивніші відправники",
                text="Packets",
            )
            st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.subheader("Топ-5 IP-адрес отримувачів")
        st.dataframe(top_receivers, use_container_width=True)
        if not top_receivers.empty:
            fig = px.bar(
                top_receivers,
                x="Destination IP",
                y="Packets",
                title="Найактивніші отримувачі",
                text="Packets",
            )
            st.plotly_chart(fig, use_container_width=True)


def render_anomalies(anomalies_df: pd.DataFrame) -> None:
    """
    Відображає секцію з аномаліями та підсвічує її попередженням.
    """
    st.subheader("Виявлення підозрілої активності")

    if anomalies_df.empty:
        st.success("Аномальної активності за заданими порогами не виявлено.")
        return

    st.error("Увага! Виявлено потенційно підозрілу мережеву активність.")
    st.dataframe(anomalies_df, use_container_width=True)


def render_raw_data(filtered_df: pd.DataFrame) -> None:
    """
    Відображає відфільтровані сирі дані.
    """
    st.subheader("Відфільтровані мережеві логи")
    st.dataframe(filtered_df, use_container_width=True)


def render_csv_instructions() -> None:
    """
    Виводить інструкцію по структурі тестового CSV-файлу.
    """
    st.markdown("---")
    st.subheader("Структура тестового CSV-файлу")
    st.markdown(
        """
        CSV-файл повинен містити такі колонки:

        - **Source IP** — IP-адреса відправника.
        - **Destination IP** — IP-адреса отримувача.
        - **Protocol** — мережевий протокол (наприклад: TCP, UDP, ICMP).
        - **Port** — номер порту.
        - **Size** — розмір пакета або запису в байтах.
        - **Time** — дата й час події.

        **Приклад одного рядка CSV:**

        ```csv
        Source IP,Destination IP,Protocol,Port,Size,Time
        192.168.1.10,8.8.8.8,TCP,443,1500,2025-05-01 10:15:30
        ```
        """
    )


def main() -> None:
    """
    Головна функція застосунку Streamlit.
    Саме тут виконується керування потоком програми.
    """
    render_header()

    # Отримуємо параметри з бічної панелі.
    uploaded_file, packet_threshold, bytes_threshold = render_sidebar()

    # Якщо файл ще не завантажено, показуємо підказку та інструкцію.
    if uploaded_file is None:
        st.info("Для початку роботи завантажте CSV-файл у бічній панелі зліва.")
        render_csv_instructions()
        return

    try:
        # Створюємо конфігурацію аномалій на основі значень, введених користувачем.
        config = AnalysisConfig(
            packet_threshold=packet_threshold,
            bytes_threshold=bytes_threshold,
        )

        # Ініціалізуємо аналізатор через фабричний метод.
        analyzer = NetworkAnalyzer.from_csv(uploaded_file, config=config)

        # Отримуємо доступні значення для фільтрації.
        available_protocols = analyzer.get_unique_protocols()
        available_ports = analyzer.get_unique_ports()
        min_time, max_time = analyzer.get_time_range()

        st.sidebar.subheader("Фільтри")

        selected_protocols = st.sidebar.multiselect(
            "Оберіть протоколи",
            options=available_protocols,
            default=available_protocols,
        )

        selected_ports = st.sidebar.multiselect(
            "Оберіть порти",
            options=available_ports,
            default=[],
        )

        ip_query = st.sidebar.text_input(
            "Пошук за IP-адресою",
            value="",
            help="Можна ввести повну IP-адресу або її фрагмент.",
        )

        start_time = None
        end_time = None

        # Якщо в наборі даних є коректний часовий інтервал, надаємо вибір користувачу.
        if min_time is not None and max_time is not None:
            start_time = st.sidebar.datetime_input("Початок періоду", value=min_time.to_pydatetime())
            end_time = st.sidebar.datetime_input("Кінець періоду", value=max_time.to_pydatetime())

        # Застосовуємо фільтри.
        filtered_df = analyzer.apply_filters(
            protocols=selected_protocols,
            ports=selected_ports,
            start_time=pd.to_datetime(start_time) if start_time else None,
            end_time=pd.to_datetime(end_time) if end_time else None,
            ip_query=ip_query,
        )

        # Якщо після фільтрації нічого не залишилось, повідомляємо користувача.
        if filtered_df.empty:
            st.warning("За заданими фільтрами дані не знайдено. Спробуйте змінити параметри.")
            render_csv_instructions()
            return

        # Обчислюємо аналітичні показники.
        metrics = analyzer.get_summary_metrics()
        top_senders = analyzer.get_top_senders(top_n=5)
        top_receivers = analyzer.get_top_receivers(top_n=5)
        protocol_distribution = analyzer.get_protocol_distribution()
        port_distribution = analyzer.get_port_distribution(top_n=10)
        anomalies = analyzer.detect_anomalies()

        # Виводимо короткі метрики.
        st.subheader("Ключові показники")
        render_metrics(metrics)

        # Блок з графіками та таблицями.
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            render_protocol_chart(protocol_distribution)

        with chart_col2:
            render_port_chart(port_distribution)

        render_ip_charts(top_senders, top_receivers)
        render_anomalies(anomalies)
        render_raw_data(filtered_df)
        render_csv_instructions()

        # Додатково дозволяємо скачати відфільтровані результати як CSV.
        csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Завантажити відфільтрований звіт у CSV",
            data=csv_data,
            file_name="network_report_filtered.csv",
            mime="text/csv",
        )

    except pd.errors.EmptyDataError:
        # Обробка конкретної помилки: файл існує, але не містить даних.
        st.error("Помилка: CSV-файл порожній. Завантажте файл з даними.")
    except KeyError as exc:
        # Обробка ситуації, коли не вистачає необхідних колонок.
        st.error(f"Помилка структури файлу: {exc}")
        render_csv_instructions()
    except Exception as exc:
        # Загальна обробка інших помилок, щоб додаток не завершився аварійно.
        st.error(f"Сталася неочікувана помилка під час аналізу даних: {exc}")


# Точка входу в програму.
# Умова гарантує, що main() виконається лише при прямому запуску файлу.
if __name__ == "__main__":
    main()
