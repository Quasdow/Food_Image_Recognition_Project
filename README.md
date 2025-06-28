# Food_Image_Recognition_Project
A deep learning-based food image recognition system using the InceptionV3 model on the Food-101 dataset, integrated into a Flask and Streamlit web application for real-time food classification and nutritional analysis. The project achieves a validation accuracy of 83.55% and a test accuracy of 83.06%, with an average inference time of 0.0205 seconds per image.

## Project Overview
This project develops a robust food image recognition system capable of classifying 101 food categories from the Food-101 dataset. The system leverages transfer learning with a fine-tuned InceptionV3 convolutional neural network (CNN), data augmentation, and comprehensive evaluation metrics (accuracy, F1-score, ROC curves). The trained model is integrated into a web application with an intuitive interface, providing real-time classification, nutritional insights, and dietary recommendations.

## Features
- **Model Performance:** Validation accuracy of 83.55%, test accuracy of 83.06%, and macro-averaged F1-score of 0.8344.
- **Real-Time Inference:** Average inference time of 0.0205 seconds per image.
- **Web Application:** Built with Flask (`app.py`) for backend processing and Streamlit (`ui.py)` for an interactive UI, displaying top-5 predictions and nutritional data.
- **Comprehensive Evaluation:** Includes confusion matrices, t-SNE visualizations, and Grad-CAM heatmaps, implemented in `evaluation.ipynb`.
- **Nutritional Analysis:** Retrieves nutritional data via the USDA API (`get_nutrition_data.py`) and displays it in tables and pie charts.

## Structure
```
Food_Image_Recognition_Project/
├── data/
│   ├── food_nutrition.csv    # Nutritional data
├── models/
│   └── history_101class.log  # Training log
├── notebooks/
│   ├── Food_Image_Recognition.ipynb  # Model training notebook
│   └── evaluation.ipynb             # Model evaluation notebook
├── src/
│   ├── app.py                # Web app backend
│   ├── ui.py                 # Web app frontend
│   ├── utils.py              # Utility functions
│   ├── config.py             # Configuration settings
│   ├── food_descriptions.py  # Food descriptions
│   └── get_nutrition_data.py # Nutrition data retrieval
├── figures/                  # Figures used in the report
├── Pre-thesis Report.pdf
├── requirements.txt
├── .env            # Example environment variables
└── README.md                 # Project overview (this file)
```

**Note**: The Food-101 dataset (`data/train/`, `data/test/`, `data/validation/`) and the trained model (`best_model_101class.hdf5`) are not included due to their large size. See **Setup** for download instructions.


## Setup
**1. Clone the repository:**
```
git clone https://github.com/Quasdow/Food_Image_Recognition_Project
cd FoodRecognitionProject
```

**2. Install dependencies:**
`pip install -r requirements.txt`

**3. Download the Food-101 dataset:**
```
Download from ETH Zurich Vision Lab or Kaggle.
Extract and place in data/train/, data/test/, and data/validation/.
```

**4. Download the trained model:**
```
Download best_model_101class.hdf5 from Google Drive.
Place in models/.
```

**5. Configure environment variables:**

- Copy .env.example to .env: `cp .env`
- Add your USDA API key to .env: `API_KEY=your_usda_api_key_here`

**6. Run the web application:**
`streamlit run src/app.py`


## Usage
1. Open the web application in your browser (default: `http://localhost:8501`).
2. Upload a food image (JPEG/PNG) via the Streamlit interface.
3. View the top-5 predicted food categories, confidence scores, nutritional facts (table and pie chart), dietary recommendations, and an optional Grad-CAM heatmap for interpretability.
4. Explore the training and evaluation process in `notebooks/Food_Image_Recognition.ipynb` and `notebooks/evaluation.ipynb`.


## Documentation
- The project report (`Pre-thesis Report.pdf`) is available upon request due to sensitive information. Contact **baduy3723@gmail.com** for access.
- Figures (e.g., confusion matrices, t-SNE visualizations) are stored in `figures/`.


## Requirements
Key dependencies (see `requirements.txt` for the full list):

- Python 3.8
- TensorFlow 2.4.0
- Keras 2.4.0
- Streamlit 1.5.0
- Flask 2.0.1
- OpenCV 4.5.1
- Scikit-learn 0.24.1
- Matplotlib 3.3.4
- Seaborn 0.11.1
- Plotly


## License
This project is licensed under the MIT License. See the LICENSE file for details.


## Acknowledgments
- Food-101 dataset provided by ETH Zurich Vision Lab.
- Computational resources from Google Colab and Kaggle.
- Open-source libraries: TensorFlow, Keras, Streamlit, Flask, and others.

## Contact
For questions or collaboration, contact me at **baduy3723@gmail.com**.
