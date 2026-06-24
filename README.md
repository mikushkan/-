# 🚀 EduNavigator: AI-Powered Career Guidance System for School Students

**EduNavigator** is an intelligent web application designed to help middle and high school students discover their ideal career paths. By leveraging machine learning algorithms and interactive data visualization, the system analyzes students' academic inclinations across various disciplines and provides objective, data-driven career recommendations.

Developed as a capstone project for the **"Orleu" National Center for Professional Development** training course (Kazakhstan).

---

## 🌟 Key Features

* **AI-Driven Predictions:** Uses an ensemble Machine Learning model (**Random Forest**) to predict the most suitable career field based on student inputs.
* **Real-Time Interactive UI:** Built with **Streamlit**, allowing students to adjust their subject interests (1-100 scale) using intuitive sliders and get instant results.
* **Probability Breakdown:** Shows the precise probability percentage for alternative career paths, giving a comprehensive view of the student's potential.
* **Descriptive Statistics & Analytics:** Features an embedded analytics dashboard with **Seaborn** data visualizations, helping educators analyze overall student trends.

---

## 🛠️ Tech Stack & Course Mapping

This project practically implements core modules covered during the professional development course:
* **Frontend/Backend:** Python, Streamlit (Web Deployment)
* **Machine Learning:** Scikit-Learn (`Random Forest Classifier` - Module 6.4)
* **Data Manipulation:** Pandas, NumPy (Module 4.1)
* **Data Visualization:** Seaborn, Matplotlib (Module 3.4)

---

## 📊 How It Works (Methodology)

1. **Input Data:** The user inputs scores (1–100) reflecting interest/performance in 5 core academic areas:
   * Technical (Math, Physics, Informatics)
   * Natural Sciences (Biology, Chemistry)
   * Social Sciences (History, Geography)
   * Humanities (Languages, Literature)
   * Creative Arts (Art, Design, Music)
2. **Model Processing:** The trained **Random Forest** model processes the feature vector.
3. **Output:** The app displays the primary recommended domain alongside a data table showing matching probabilities for other fields.

---

## 🚀 Installation & Local Setup

To run this project locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mikushkan/-.git](https://github.com/mikushkan/-.git)
   cd -
