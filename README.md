# <p align="center">🚀 FaceRecognition Eigenface with Streamlit</p>

<p align="center">
  <a href="[URL_DEMO]" target="_blank">🔍 Live Demo Program</a> ·
</p>

---

## 📖 Brief Description

This project implements **Eigenface-based Face Recognition** in **Python** with an interactive **Streamlit** web interface. You can upload a photo, preview preprocessing steps, and perform real-time face recognition right in your browser. 

---

## 👥 Team Members

| Name                 | NIM / ID        |
| -------------------  | --------------- |
| Mufti Faris Murtadho | L0124133        |
| Yashif Victoriawan   | L0124124        |
| Yusran Rizqi Laksono | L0124125        |

---

## 🧰 Technologies Used

- **Language**: Python 3.9+  
- **Libraries & Frameworks**:  
  - OpenCV  
  - NumPy  
  - scikit-learn  
  - Streamlit  
  - Pillow  
- **Tools & Platforms**:  
  - Git & GitHub  
  - GitHub Actions (CI/CD)

---

## 📁 Struktur Proyek

```text
[repo-root]/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── main.py
│   ├── face_recognition.py
│   ├── dataset_loader.py
│   └── utils.py
├── data/
│   ├── images/
│   └── models/
└── docs/
    ├── architecture.md
    └── user_guide.md
```

---

## ⚙️ Setup Instructions

You can setup your project by cloning this repository and install the libraries above.

1. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate      # Windows

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   Or, manually install each package:
   ```bash
   pip install opencv-python
   pip install numpy
   pip install scikit-learn
   pip install streamlit
   pip install pillow
   ```
   💡 Make sure you're using Python 3.9 or later to avoid compatibility issues.

3. How to Run the App
   ```bash
   streamlit run main.py
   ```
   You will see the web interface where you can:
   Upload a face image
   View preprocessing steps
   Get prediction results using Eigenface algorithm

---

## 📬 Contact
“I hope this project helps you grasp eigenface concepts—feel free to send feedback!”

💼 LinkedIn: Mufti Faris Murtadho(https://="www.linkedin.com/in/mufti-faris)

---

## NOTE
For the contents of the "test" folder it is necessary to download from https://www.kaggle.com/datasets/hereisburak/pins-face-recognition , which contains a downloadable face database.
