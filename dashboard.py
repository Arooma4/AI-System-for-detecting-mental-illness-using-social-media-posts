
import streamlit as st
import pandas as pd
import os, sys, time
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests

import plotly.io as pio
pio.templates.default = "plotly_dark"


# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Mental Health System", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    font-family: 'Segoe UI', sans-serif;
}

.glass {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 20px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    margin-bottom: 25px;
    transition: all 0.3s ease-in-out;
}

.glass:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 15px 50px rgba(0,0,0,0.6);
    border: 1px solid #00F5A0;
}

.fade-in {
    animation: fadeIn 0.6s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

h1, h2 {
    text-shadow: 0 0 10px rgba(0,255,200,0.3);
}

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(8px);
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)
# ---------------- SAFE LOTTIE ----------------
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

lottie = load_lottie("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
from predict import predict_with_confidence

# ---------------- DATA ----------------
df = pd.read_csv(os.path.join(BASE_DIR,"data","final_dataset_small.csv"))
if "text" not in df.columns:
    df["text"] = df.astype(str).apply(" ".join, axis=1)




# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Navigation",[
"🏠 Home",
"🧠 Prediction",
"🤖 Chatbot",
"🧠 Disease Insights",
"📊 Analytics",
"📈 Trends",
"📌 About"
])

# ---------------- CHATBOT TYPING FUNCTION ----------------
def type_text(text):
    placeholder = st.empty()
    for i in range(len(text)+1):
        placeholder.markdown(text[:i])
        time.sleep(0.01)

# ---------------- HOME ----------------
if page=="🏠 Home":

    if lottie:
        st_lottie(lottie,height=250)
    else:
        st.info("✨ Welcome to AI Mental Health System")

    st.title("🧠 AI Mental Health Detection System")

    # KPI CARDS (ADDED)
    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class="glass">
    <h4>📊 Total Records</h4>
    <h2>{len(df)}</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="glass">
    <h4>🧠 Conditions</h4>
    <h2>{df["label"].nunique()}</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="glass">
    <h4>📈 Avg Text Length</h4>
    <h2>{int(df["text"].apply(len).mean())}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.image("https://images.pexels.com/photos/4101143/pexels-photo-4101143.jpeg",
             use_container_width=True)

    st.markdown("""
### 🌍 Why This Matters
Mental health affects millions worldwide. Early detection helps prevent serious conditions.

### 🚀 Features
- AI Prediction  
- Chatbot Support  
- Disease Awareness  
- Data Analytics  
""")
    st.markdown('</div>', unsafe_allow_html=True)
  

# ---------------- PREDICTION ----------------
if page=="🧠 Prediction":

    st.title("🧠 Analyze Text")

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    text = st.text_area("Enter text")

    if st.button("Analyze"):

        if text.strip() == "":
            st.warning("Please enter some text")
        else:
            with st.spinner("🧠 AI is understanding your emotions..."):
                try:
                    label, confidence, probs = predict_with_confidence(text)
                except:
                    label, confidence, probs = "Error", 0, {}

            # 🎯 PREDICTION RESULT
            st.markdown(f"""
            <div class="glass">
            <h2>🧠 Prediction: {label}</h2>
            </div>
            """, unsafe_allow_html=True)

            # 📊 CONFIDENCE
            st.progress(int(confidence))

            st.markdown(f"""
            <div class="glass fade-in">
            <b>🧠 AI Confidence Level:</b> {confidence}%
            </div>
            """, unsafe_allow_html=True)

            # 🎯 COLOR FEEDBACK
            if confidence > 80:
                st.success(f"High Confidence: {confidence}%")
            elif confidence > 50:
                st.warning(f"Medium Confidence: {confidence}%")
            else:
                st.error(f"Low Confidence: {confidence}%")

            # 📈 GAUGE CHART
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Confidence"},
                gauge={'axis': {'range':[0,100]}}
            ))
            st.plotly_chart(fig,use_container_width=True)

            # 📊 PROBABILITY CHART
            if probs:
                prob_df = pd.DataFrame(probs.items(),columns=["Condition","Probability"])
                st.plotly_chart(px.bar(prob_df,x="Condition",y="Probability",color="Condition"))

            # 🔍 KEYWORDS
            from collections import Counter
            words = text.split()
            common = Counter(words).most_common(5)

            st.markdown("### 🔍 Key Words Detected")
            for w,c in common:
                st.write(f"{w} → {c}")

            # 🧠 WHY THIS PREDICTION (INSIDE BUTTON)
            st.markdown("""
            <div class="glass fade-in">
            <h3>🧠 Why this prediction?</h3>
            The model analyzes emotional keywords and patterns such as:
            - worry / stress → anxiety  
            - sadness / emptiness → depression  
            - mood swings → bipolar  
            </div>
            """, unsafe_allow_html=True)

            # 💡 SUGGESTED ACTIONS (INSIDE BUTTON)
            if label.lower() == "anxiety":
                st.info("💡 Try breathing exercises, reduce stress, and talk to someone.")
            elif label.lower() == "depression":
                st.info("💡 Maintain routine, exercise, and consider professional help.")
            elif label.lower() == "bipolar":
                st.info("💡 Track mood patterns and consult a specialist.")

    st.markdown('</div>', unsafe_allow_html=True)
  
# ---------------- CHATBOT ----------------
if page=="🤖 Chatbot":

    st.title("🤖 AI Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages=[{
            "role":"assistant",
            "content":"""
👋 Hello! I’m your AI Mental Health Assistant.

You can ask:
- What is depression?
- I feel anxious
- Suggest remedies
- General questions

I provide support + AI insights 💙
"""
        }]

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything...")

    if user_input:

        st.session_state.messages.append({"role":"user","content":user_input})

        with st.chat_message("assistant"):

            text=user_input.lower()

            if "bpd" in text:
                reply="BPD involves emotional instability and fear of abandonment."

            elif "depression" in text:
                reply="Depression is persistent sadness and low interest."

            elif "anxiety" in text:
                reply="Anxiety involves excessive worry and stress."

            elif "remedies" in text:
                reply="Therapy, meditation, exercise, and support help."

            elif any(w in text for w in ["sad","lonely"]):
                reply="I'm really sorry you're feeling this way 💙"
            elif "help" in text or "what should i do" in text:
                reply = "Try talking to someone you trust, maintain routine, and consider professional help 💙"

            else:
                try:
                    label,confidence,_=predict_with_confidence(user_input)

                    if confidence>70:
                        reply=f"It seems like {label}. I'm here for you 💙"
                    else:
                        reply="Hello!!How can I assist you? 😊"
                except:
                    reply="Hello!!How can I assist you?😊"

            type_text(reply)   # ADDED typing effect

        st.session_state.messages.append({"role":"assistant","content":reply})

    st.markdown('</div>', unsafe_allow_html=True)
  
# ---------------- DISEASE INSIGHTS ----------------
if page=="🧠 Disease Insights":

    st.markdown("""
    <h1 style='text-align:center; color:#00F5A0;'>
    🧠 Mental Health Knowledge Hub
    </h1>
    """, unsafe_allow_html=True)

    search = st.text_input("🔍 Search disease")

    if search:
        st.success(f"Showing results for: {search}")
    

    diseases = {

        "Depression": {
            "img": "https://images.pexels.com/photos/3771089/pexels-photo-3771089.jpeg",
            "content": """
### 📊 Overview  
Depression is a common mental disorder that negatively affects how you feel, think, and act.

### 🔍 Causes  
- Genetics  
- Trauma / stress  
- Brain chemistry imbalance  

### ⚠️ Symptoms  
- Persistent sadness  
- Loss of interest  
- Fatigue  
- Sleep issues  

### 🧩 Types  
- Major Depressive Disorder  
- Persistent Depressive Disorder  
- Seasonal Affective Disorder  

### 💊 Treatment  
- Therapy (CBT)  
- Medication  
- Exercise & lifestyle changes  
"""
        },

        "Anxiety": {
            "img": "https://images.pexels.com/photos/4101143/pexels-photo-4101143.jpeg",
            "content": """
### 📊 Overview  
Anxiety involves excessive worry, fear, and nervousness affecting daily life.

### 🔍 Causes  
- Genetics  
- Stressful life events  
- Personality traits  

### ⚠️ Symptoms  
- Overthinking  
- Restlessness  
- Rapid heartbeat  

### 🧩 Types  
- Generalized Anxiety Disorder  
- Panic Disorder  
- Social Anxiety  

### 💊 Treatment  
- Meditation  
- Therapy  
- Stress management  
"""
        },

        "Bipolar": {
            "img": "https://images.pexels.com/photos/3772618/pexels-photo-3772618.jpeg",
            "content": """
### 📊 Overview  
Bipolar disorder is characterized by extreme mood swings.

### 🔍 Causes  
- Genetic factors  
- Brain structure changes  

### ⚠️ Symptoms  
- Mania (high energy)  
- Depression (low energy)  

### 🧩 Types  
- Bipolar I  
- Bipolar II  
- Cyclothymia  

### 💊 Treatment  
- Medication  
- Therapy  
- Mood tracking  
"""
        },

        "BPD": {
            "img": "https://images.pexels.com/photos/4101140/pexels-photo-4101140.jpeg",
            "content": """
### 📊 Overview  
Borderline Personality Disorder involves emotional instability and fear of abandonment.

### 🔍 Causes  
- Childhood trauma  
- Emotional neglect  
- Genetics  

### ⚠️ Symptoms  
- Intense emotions  
- Fear of abandonment  
- Impulsive behavior  

### 🧩 Types  
- Impulsive  
- Discouraged  
- Self-destructive  

### 💊 Treatment  
- DBT therapy  
- Emotional regulation training  
"""
        },

        "Schizophrenia": {
            "img": "https://images.pexels.com/photos/4101142/pexels-photo-4101142.jpeg",
            "content": """
### 📊 Overview  
Schizophrenia is a serious mental disorder affecting thinking and perception.

### 🔍 Causes  
- Genetics  
- Brain chemistry imbalance  
- Environmental triggers  

### ⚠️ Symptoms  
- Hallucinations  
- Delusions  
- Disorganized thinking  

### 🧩 Types  
- Paranoid  
- Disorganized  
- Catatonic  

### 💊 Treatment  
- Antipsychotic medication  
- Therapy  
- Social support  
"""
        }
    }

    # GRID UI (FIXED PROPERLY)
    cols = st.columns(2)

    for i, (d, info) in enumerate(diseases.items()):

        if search.lower() in d.lower():

            with cols[i % 2]:
                st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
                st.markdown(f"""
                <h2 style="color:#00F5A0;">🧠 {d}</h2>
                <hr style="border:1px solid rgba(255,255,255,0.2)">
                """, unsafe_allow_html=True)
                try:
                    st.image(info["img"], use_container_width=True)
                except:
                    st.warning("Image not available")
                st.markdown(info["content"])
                st.markdown('</div>', unsafe_allow_html=True)
        if search and not any(search.lower() in d.lower() for d in diseases):
            st.warning("No matching disease found")
   


# ---------------- ANALYTICS ----------------
if page=="📊 Analytics":

    st.title("📊 Data Analytics")

    # 🔹 HEADER CARD
    st.markdown("""
    <div class="glass fade-in">
    <h3>📊 Insights Overview</h3>
    This section shows patterns from dataset + real-world mental health trends.
    </div>
    """, unsafe_allow_html=True)

    # 📊 DATA PREP (UNCHANGED)
    dist = df["label"].value_counts().reset_index()
    dist.columns = ["Condition", "Count"]

    # 🔥 DATASET INFO (KEPT BUT CLARIFIED)
    top = df["label"].value_counts().idxmax()

    st.markdown(f"""
    <div class='glass'>
    🔥 Most common in dataset: <b>{top}</b>
    </div>
    """, unsafe_allow_html=True)

    # 🌍 REAL-WORLD INSIGHTS (NEW - IMPORTANT)
    st.markdown("""
    <div class="glass fade-in">
    <h3>🌍 Global Mental Health Insights</h3>

    ✔ Depression affects over <b>280 million people</b> worldwide  
    ✔ Anxiety disorders affect around <b>300 million people</b>  
    ✔ Bipolar disorder affects about <b>40–50 million people</b>  
    ✔ Schizophrenia affects about <b>24 million people</b>  
    ✔ Mental health issues are increasing globally due to stress and lifestyle  

    </div>
    """, unsafe_allow_html=True)

    # 📊 TABS
    tab1, tab2 = st.tabs(["📊 Charts", "☁️ WordCloud"])

    # ---------------- CHARTS ----------------
    with tab1:

        # 🎨 BAR CHART
        fig_bar = px.bar(
            dist,
            x="Condition",
            y="Count",
            color="Condition",
            text="Count",
            color_discrete_sequence=px.colors.sequential.Tealgrn
        )

        fig_bar.update_traces(textposition="outside")

        fig_bar.update_layout(
            title="📊 Dataset Distribution",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 🥧 PIE CHART
        fig_pie = px.pie(
            dist,
            names="Condition",
            values="Count",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Aggrnyl
        )

        fig_pie.update_layout(
            title="🧠 Dataset Proportion",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- WORD CLOUD ----------------
    with tab2:

        wc = WordCloud(
            width=1000,
            height=500,
            background_color="black",
            colormap="viridis",
            contour_width=2,
            contour_color="white"
        ).generate(" ".join(df["text"]))

        fig_wc, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")

        st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
        st.pyplot(fig_wc)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TEXT LENGTH ANALYSIS ----------------
    df["length"] = df["text"].apply(len)

    fig_len = px.histogram(
        df,
        x="length",
        nbins=30,
        title="📏 Text Length Distribution",
        color_discrete_sequence=["#00F5A0"]
    )

    fig_len.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
    st.plotly_chart(fig_len, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- GLOBAL COMPARISON CHART (NEW) ----------------
    st.markdown("### 🌍 Real-World Comparison")

    real_data = pd.DataFrame({
        "Condition": ["Anxiety", "Depression", "Bipolar", "Schizophrenia", "BPD"],
        "Affected (Millions)": [300, 280, 45, 24, 20]
    })

    fig_real = px.bar(
        real_data,
        x="Condition",
        y="Affected (Millions)",
        color="Condition",
        text="Affected (Millions)",
        color_discrete_sequence=px.colors.sequential.Plasma
    )

    fig_real.update_layout(
        title="🌍 Global Mental Health Distribution",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
    st.plotly_chart(fig_real, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- EXPLANATION ----------------
    st.markdown("""
    <div class="glass fade-in">
    <h3>📈 What do these charts show?</h3>

    - Dataset charts → patterns in collected data  
    - Word cloud → frequently used words  
    - Histogram → text length behavior  
    - Global chart → real-world mental health statistics  

    This combination provides both technical analysis and real-world understanding.

    </div>
    """, unsafe_allow_html=True)
    

# ---------------- TRENDS ----------------
if page=="📈 Trends":

    st.title("📈 Trends Analysis")

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.write("Shows real-world mental health trends over time based on global estimates.")

    # 🌍 REAL-WORLD TREND DATA
    trend_data = pd.DataFrame({
        "Year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        "Anxiety": [230, 235, 240, 250, 260, 280, 290, 295, 300],
        "Depression": [250, 255, 260, 265, 270, 290, 300, 305, 310],
        "Bipolar": [40, 41, 42, 43, 44, 45, 46, 47, 48],
        "Schizophrenia": [21, 21.5, 22, 22.5, 23, 23.5, 24, 24.2, 24.5]
    })

    # 🔄 Convert to long format
    trend_long = trend_data.melt(
        id_vars="Year",
        var_name="Condition",
        value_name="Cases (Millions)"
    )

    # 🎛️ YEAR RANGE SLIDER (NEW)
    year_range = st.slider(
        "📅 Select Year Range",
        min_value=int(trend_data["Year"].min()),
        max_value=int(trend_data["Year"].max()),
        value=(2018, 2023)
    )

    filtered = trend_long[
        (trend_long["Year"] >= year_range[0]) &
        (trend_long["Year"] <= year_range[1])
    ]

    # 🎯 CONDITION FILTER (NEW)
    conditions = st.multiselect(
        "🧠 Select Conditions",
        options=trend_long["Condition"].unique(),
        default=trend_long["Condition"].unique()
    )

    filtered = filtered[filtered["Condition"].isin(conditions)]

    # 📈 TREND LINE CHART
    fig_line = px.line(
        filtered,
        x="Year",
        y="Cases (Millions)",
        color="Condition",
        markers=True,
        line_shape="spline"
    )

    fig_line.update_traces(line=dict(width=4))

    fig_line.update_layout(
        title="🌍 Global Mental Health Trends Over Time",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend_title="Condition"
    )

    st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 📊 INSIGHT BOX
    st.markdown("""
    <div class="glass fade-in">
    <h3>📈 Key Trend Insights</h3>

    ✔ Mental health cases have steadily increased over time  
    ✔ Significant rise observed after 2020 (pandemic impact)  
    ✔ Anxiety and depression show highest growth trends  
    ✔ Bipolar and schizophrenia remain relatively stable  

    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
   

# ---------------- ABOUT ----------------
if page=="📌 About":

    st.title("📌 About Project")

    # 🌟 INTRO
    st.markdown("""
    <div class="glass fade-in">
    <h2>🧠 AI Mental Health Detection System</h2>
    This project uses Artificial Intelligence and Natural Language Processing (NLP) 
    to analyze user text and detect potential mental health conditions.
    It also provides insights, chatbot support, and real-world analytics.
    </div>
    """, unsafe_allow_html=True)

    # 🎯 OBJECTIVE
    st.markdown("""
    <div class="glass fade-in">
    <h3>🎯 Objective</h3>
    - Detect mental health conditions from text  
    - Provide early awareness and insights  
    - Assist users with chatbot-based support  
    - Visualize trends using analytics dashboard  
    </div>
    """, unsafe_allow_html=True)

    # ⚙️ TECH STACK
    st.markdown("""
    <div class="glass fade-in">
    <h3>⚙️ Tech Stack</h3>

    ✔ Python  
    ✔ Machine Learning (Text Classification)  
    ✔ Natural Language Processing (NLP)  
    ✔ Streamlit (Frontend UI)  
    ✔ Plotly (Visualization)  

    </div>
    """, unsafe_allow_html=True)

    # 🔄 WORKFLOW
    st.markdown("""
    <div class="glass fade-in">
    <h3>🔄 Execution Flow</h3>

    1️⃣ User enters text  
    ↓  
    2️⃣ NLP preprocessing  
    ↓  
    3️⃣ Model prediction  
    ↓  
    4️⃣ Confidence score + insights  
    ↓  
    5️⃣ Visualization & recommendations  

    </div>
    """, unsafe_allow_html=True)

    # 🤖 FEATURES
    st.markdown("""
    <div class="glass fade-in">
    <h3>🚀 Key Features</h3>

    ✔ AI-based mental health prediction  
    ✔ Explainable AI (why prediction)  
    ✔ Chatbot for interaction  
    ✔ Disease knowledge system  
    ✔ Data analytics dashboard  
    ✔ Real-world trend analysis  

    </div>
    """, unsafe_allow_html=True)

    # 🌍 REAL-WORLD IMPACT
    st.markdown("""
    <div class="glass fade-in">
    <h3>🌍 Real-World Impact</h3>

    Mental health issues are increasing globally.  
    This system helps in early detection and awareness,  
    which can lead to better mental health support and prevention.

    </div>
    """, unsafe_allow_html=True)

    # ⚠️ LIMITATIONS
    st.markdown("""
    <div class="glass fade-in">
    <h3>⚠️ Limitations</h3>

    - Not a medical diagnosis tool  
    - Accuracy depends on dataset quality  
    - Limited to text-based input  
    - Chatbot is rule-based (not full AI)  

    </div>
    """, unsafe_allow_html=True)

    # 🔮 FUTURE SCOPE
    st.markdown("""
    <div class="glass fade-in">
    <h3>🔮 Future Enhancements</h3>

    - Integration with ChatGPT / LLM APIs  
    - Voice-based input system  
    - Real-time data collection  
    - More accurate models (BERT / Transformers)  
    - Mobile application  

    </div>
    """, unsafe_allow_html=True)

    # 👩‍💻 AUTHOR
    st.markdown("""
    <div class="glass fade-in">
    <h3>👩‍💻 Developed By</h3>

    Arooma Kashyap,
    Ruchi Pal,
    Anshu Vikal,
    Dimple Negi

    </div>
    """, unsafe_allow_html=True)






