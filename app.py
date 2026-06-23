import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 1. СТАТИСТИКА ЖӘНЕ ДЕРЕКТЕР КӨЗІ (Dataset)
# Тәжірибе үшін қолданбалы деректер жиынтығын құрастырамыз (Модуль 4.1 талабы бойынша)
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 200
    
    # Оқушылардың пәндер бойынша бейімділігі (1-ден 100-ге дейін ұпай)
    data = {
        'Math_Physics': np.random.randint(50, 100, n_samples),
        'Biology_Chemistry': np.random.randint(40, 100, n_samples),
        'History_Geography': np.random.randint(40, 100, n_samples),
        'Literature_Languages': np.random.randint(40, 100, n_samples),
        'Creativity_Art': np.random.randint(30, 100, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Ең жоғары ұпайға байланысты мамандық бағытын (таргет) анықтау
    targets = []
    for idx, row in df.iterrows():
        max_col = row.idxmax()
        if max_col == 'Math_Physics':
            targets.append('IT және Инженерия')
        elif max_col == 'Biology_Chemistry':
            targets.append('Медицина және Био can')
        elif max_col == 'History_Geography':
            targets.append('Құқық, Тарих және География')
        elif max_col == 'Literature_Languages':
            targets.append('Филология және Педагогика')
        else:
            targets.append('Өнер және Дизайн')
            
    df['Target_Profession'] = targets
    return df

df = load_data()

# 2. МАШИНАЛЫҚ ОҚЫТУ МОДЕЛІ (Модуль 6.4 - Random Forest)
X = df.drop('Target_Profession', axis=1)
y = df['Target_Profession']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. ВЕБ-ИНТЕРФЕЙС (STREAMLIT) ТАҢБАЛАУЫ
st.set_page_config(page_title="EduNavigator - Кәсіптік бағдар беру", layout="wide")

st.title("🚀 EduNavigator: Мектеп оқушыларына арналған интеллектуалды кәсіптік бағдар беру жүйесі")
st.write("Бұл жүйе машиналық оқыту (Random Forest) алгоритмі негізінде оқушының бейімділігін талдап, мамандық ұсынады.")

# Сол жақ панель: Деректерді енгізу
st.sidebar.header("📋 Оқушының бейімділік көрсеткіштері (1-100 ұпай):")
math_phys = st.sidebar.slider("Техникалық бағыт (Математика/Физика/Информатика):", 1, 100, 75)
bio_chem = st.sidebar.slider("Жаратылыстану (Биология/Химия):", 1, 100, 50)
hist_geo = st.sidebar.slider("Қоғамдық бағыт (Тарих/География):", 1, 100, 40)
lit_lang = st.sidebar.slider("Гуманитарлық бағыт (Тілдер/Әдебиет):", 1, 100, 60)
creative = st.sidebar.slider("Шығармашылық (Өнер/Дизайн/Музыка):", 1, 100, 30)

# Негізгі бетті екі бағанға бөлу
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔮 Машиналық оқыту моделінің болжамы")
    
    # Пайдаланушы енгізген деректен массив жасау
    user_data = np.array([[math_phys, bio_chem, hist_geo, lit_lang, creative]])
    
    # Болжам жасау (Prediction)
    prediction = model.predict(user_data)[0]
    probabilities = model.predict_proba(user_data)[0]
    
    st.success(f"Ұсынылатын негізгі бағыт: **{prediction}**")
    
    # Пайыздық сәйкестікті көрсету (Модуль 2.2 - Ықтималдықтар)
    st.write("### Басқа бағыттармен сәйкестік ықтималдығы:")
    prob_df = pd.DataFrame({
        'Бағыттар': model.classes_,
        'Ықтималдық (%)': np.round(probabilities * 100, 2)
    }).sort_values(by='Ықтималдық (%)', ascending=False)
    
    st.dataframe(prob_df, use_container_width=True)

with col2:
    st.subheader("📊 Модельді оқытуға арналған деректер анализі")
    # Модуль 3.4 бойынша Seaborn визуализациясы
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, y='Target_Profession', palette='Set2', ax=ax)
    plt.title("Деректер қорындағы мамандықтардың үлесі")
    plt.xlabel("Оқушы саны")
    plt.ylabel("")
    st.pyplot(fig)

# Төменгі бөлім: Статистикалық қорытынды (Модуль 2.4/4.1 талабы)
st.markdown("---")
st.subheader("📈 Жалпы база бойынша сипаттамалық статистика (Descriptive Statistics)")
if st.checkbox("Базадағы оқушылардың орташа көрсеткіштерін көру"):
    st.write(df.describe())