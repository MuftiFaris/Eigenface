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

## 📁 Structure Project

```text
Eigenface/
├── requirements.txt
├── src/
│   ├── main.py
│   ├── preprocessing.py
│   ├── distance.py
│   ├── __init__.py
│   ├── eigenface.py
│   ├── utils.py
│   └── __pycache__/
│       ├── preprocessing.cpython-312.pyc
│       ├── distance.cpython-312.pyc
│       ├── eigenface.cpython-312.pyc
│       └── utils.cpython-312.pyc
├── test/
│   ├── images1 /
│   │   ├── img1.1 
│   │   ├── img1.2
│   │   ├── img1.3
│   │   ├── img1.4
│   │   └── img1....
│   ├── images2 /
│   │   ├── img2.1 
│   │   ├── img2.2
│   │   ├── img2.3
│   │   ├── img2.4
│   │   └── img2....
│   ├── images3 /
│   │   ├── img3.1 
│   │   ├── img3.2
│   │   ├── img3.3
│   │   ├── img3.4
│   │   └── img3....
│   ├── images4 /
│   │   ├── img4.1 
│   │   ├── img4.2
│   │   ├── img4.3
│   │   ├── img4.4
│   │   └── img4....
│   ├── images5 /
│   │   ├── img5.1 
│   │   ├── img5.2
│   │   ├── img5.3
│   │   ├── img5.4
│   │   └── img5....
│   └── images../
│       ├── img...1 
│       ├── img...2
│       ├── img...3
│       ├── img...4
│       └── img......
└── docs/
    └── README.md 
```
The images and img folders are just examples, use the folders according to your path that contains the photo files.

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

## 👤 Author
“I hope this project helps you grasp eigenface concepts—feel free to send feedback!”

💼 LinkedIn: [Mufti Faris Murtadho - LinkedIn](https://www.linkedin.com/in/mufti-faris/)</br>
📍 Informatika 2024, Universitas Sebelas Maret  

---

## NOTE
For the contents of the "test" folder it is necessary to download from https://www.kaggle.com/datasets/hereisburak/pins-face-recognition , which contains a downloadable face database.
