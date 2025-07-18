This project focuses on skin cancer detection using deep learning techniques. The aim is to classify dermatoscopic images of skin moles into **benign** or **malignant** categories. 
This README provides an overview of the project structure, tools and libraries used, and step-by-step usage instructions.


🧰 Libraries Used

This project utilizes several Python libraries for deep learning, data processing, and visualization:

- **NumPy**: For numerical operations and array manipulation  
- **Pandas**: For loading and managing structured data  
- **Matplotlib**: For creating static and interactive plots  
- **Seaborn**: For statistical data visualization  
- **TensorFlow/Keras**: For building and training CNN models  
- **Scikit-learn**: For model evaluation metrics (accuracy, precision, recall, F1-score)  
- **OpenCV**: For basic image manipulation (if required)  
- **os**: For managing files and paths


📁 Data Preparation

1. Download the **ISIC 2019 dataset** and place the images and metadata into the `data/` directory.
2. Ensure that your dataset includes labeled classes (benign and malignant).
3. Perform any necessary image preprocessing (resizing, normalization) using the provided `preprocessing.py` script.


🔄 Data Preprocessing

Run the `data_preprocessing.py` or preprocessing code section in the notebook to clean and prepare the image data. This includes:

- Resizing images to match input shape of models  
- Normalizing pixel values  
- Data augmentation (flip, rotate, zoom) to handle class imbalance  
- Splitting data into training, validation, and test sets


🤖 Model Training

Use the `train_model.py` script or Jupyter Notebook to train the models. This includes:

- Defining CNN architectures: **Xception**, **EfficientNetB4**, **EfficientNetV2S**  
- Applying callbacks like EarlyStopping and ModelCheckpoint  
- Training the models on the preprocessed dataset  
- Saving the best model weights for evaluation


## 📊 Evaluation and Results

Use `evaluate_model.py` or the notebook to evaluate the model. This script includes:

- Calculating evaluation metrics: **Accuracy**, **Precision**, **Recall**, **F1-score**  
- Displaying confusion matrix and classification report  
- Plotting training vs validation loss/accuracy curves


📈 Visualization

Visualizations can be generated using `visualization.py` or in the notebook:

- Plot training history (accuracy/loss)  
- Display sample predictions with labels  
- Confusion matrix heatmaps for model performance  
- Image augmentation previews


🙏 Acknowledgments

- ISIC Archive for providing the skin lesion dataset  
- TensorFlow and Keras for deep learning support  
- Scikit-learn for evaluation metrics  
- Matplotlib and Seaborn for visualization  
- GITAM University for project guidance and academic support
