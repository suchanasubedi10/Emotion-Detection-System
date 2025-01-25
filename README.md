# Emotion-Detection-System

This repository contains the implementation of an emotion recognition model using the VGG16 architecture. The project involves fine-tuning a pre-trained VGG16 model  for emotion classification tasks using FER2013 dataset, and the necessary code is provided.

## Contents of the Repository

1. **`Vggmain.ipynb`**: The main notebook for setting up the VGG16 model, training, and evaluating it on the dataset.
2. **`try 2 vgg16 fine tune.ipynb`**: A notebook showcasing fine-tuning techniques applied to the VGG16 model to improve performance.


## Key Features

- Fine-tuning of the VGG16 model  using FER2013 dataset for emotion recognition.
- Includes trained model weights to save training time.
- Reproducible code for training and evaluation.

## Requirements

To run the notebooks and use the model, you need the following:

- Python 3.7 or later
- TensorFlow (>=2.0)
- Keras
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib

Install the dependencies using:
```bash
pip install tensorflow keras numpy pandas matplotlib
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/emotion-recognition-vgg16.git
   cd emotion-recognition-vgg16
   ```

2. Open the notebooks in Jupyter:
   ```bash
   jupyter notebook Vggmain.ipynb
   ```

3. To fine-tune the model, open and run the `try 2 vgg16 fine tune.ipynb` notebook.


## Model Details

- **Architecture**: VGG16 (Pre-trained on ImageNet)
- **Dataset**: Custom dataset for emotion recognition (not included; you can replace it with your own dataset).
- **Output**: Emotion classes such as Happy, Sad, Angry, Neutral, etc.

## Results

The fine-tuned model achieved the following accuracy on the test dataset:
- Accuracy: *67%*  
- Loss: *0.87*


## How to Use the Pre-Trained Model

You can load the model weights as follows:
```python
from tensorflow.keras.models import load_model


## Contribution

Feel free to contribute to this repository by creating issues or submitting pull requests. Contributions can include:
- Improving the documentation
- Adding more datasets
- Enhancing the model architecture
- Fine-tuning techniques

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

[Suchana Subedi]  
LinkedIn:  https://www.linkedin.com/in/suchana-subedi-33810b286 

