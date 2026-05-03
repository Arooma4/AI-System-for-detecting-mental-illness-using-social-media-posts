🧠 AI Mental Health Detection System

An AI-powered system that detects potential mental health conditions from social media posts using Natural Language Processing (NLP) and Machine Learning. The system provides predictions, confidence scores, and interactive visual analytics through a Streamlit dashboard.

📌 Overview

Mental health issues such as depression, anxiety, and bipolar disorder are increasing globally. This project analyzes textual data from social media and predicts possible mental health conditions using a trained neural network model.

🚀 Features
🧠 AI-based mental health prediction
📊 Confidence score & probability visualization
📈 Interactive analytics dashboard
☁️ WordCloud visualization
🤖 Basic chatbot support
📉 Dataset insights & trends
🛠️ Tech Stack
Programming Language: Python
Machine Learning: TensorFlow (Neural Network)
NLP: TF-IDF Vectorization
Frontend: Streamlit
Visualization: Plotly, Matplotlib, WordCloud
Libraries: Pandas, NumPy, Scikit-learn, Joblib
📂 Project Structure
mental-health-cnn-project/
│
├── data/
│   ├── raw/                          # Original datasets
│   └── final_dataset_small.csv       # Processed dataset
│
├── models/
│   ├── cnn_model.h5                  # Trained model
│   ├── vectorizer.pkl               # TF-IDF vectorizer
│   └── label_encoder.pkl            # Label encoder
│
├── src/
│   ├── create_small_dataset.py      # Dataset creation
│   ├── train_model.py               # Model training
│   └── predict.py                  # Prediction logic
│
├── dashboard.py                    # Streamlit app
└── README.md
⚙️ Installation
1. Clone Repository
git clone https://github.com/your-username/mental-health-cnn-project.git
cd mental-health-cnn-project
2. Install Dependencies
pip install pandas numpy tensorflow scikit-learn joblib streamlit plotly matplotlib wordcloud requests
▶️ How to Run
Step 1: Create Dataset
python src/create_small_dataset.py
Step 2: Train Model
python src/train_model.py
Step 3: Run Dashboard
streamlit run dashboard.py

Then open in browser:

http://localhost:8501
🔄 Workflow
User Input Text
        ↓
Text Preprocessing
        ↓
TF-IDF Feature Extraction
        ↓
Neural Network Model
        ↓
Prediction + Confidence
        ↓
Dashboard Visualization
🧠 Model Details
Model Type: Feedforward Neural Network (Dense Layers)
Input: TF-IDF features (1000 dimensions)
Architecture:
Input Layer
Dense (128 neurons)
Dense (64 neurons)
Output (Softmax)
📊 Evaluation
Metric: Accuracy (Sparse Categorical Accuracy)
Loss Function: Cross-Entropy
Optimizer: Adam
📈 Analytics

The dashboard provides:

Dataset distribution (Bar & Pie charts)
WordCloud visualization
Text length analysis
Real-world mental health trends
⚠️ Limitations
Not a medical diagnosis tool
Depends on dataset quality
Basic NLP (TF-IDF instead of deep embeddings)
Chatbot is rule-based
🔮 Future Scope
Use advanced models (BERT, Transformers)
Real-time social media integration
Mobile application
AI-powered conversational chatbot
👩‍💻 Contributors
Arooma Kashyap
Ruchi Pal
Anshu Vikal
Dimple Negi
📜 License

This project is for academic and educational purposes.

⭐ Acknowledgment
Open-source datasets
TensorFlow & Scikit-learn libraries
Streamlit for UI
