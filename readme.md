# 🤖 LSTM Text Generator 

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)  
[![GitHub license](https://img.shields.io/badge/License-MIT-green)](LICENSE)  

> Generate human-like text using LSTM & Bidirectional LSTM models. CPU-friendly version included.

---

## 🚀 Features

- 🧠 Train your own text generator from a text dataset  
- 🔄 Bidirectional LSTM architecture for better context understanding  
- 💾 Save & load best models (`best_text_generator.h5`)  
- 🖥 CPU-friendly version for machines without GPU  


---

## 🧰 Requirements

- Python 3.8+  
- TensorFlow 2.x  
- NumPy  
- scikit-learn  

Install dependencies:

```bash
pip install tensorflow numpy scikit-learn

```

---

## 🏗 How It Works

Load Dataset

Reads content.txt and converts all text to lowercase

Removes punctuation & line breaks

Optionally reduces dataset size for CPU-friendly training

Tokenization & Sequences

Uses Keras Tokenizer to map words to numbers

Creates sequences of length SEQ_LEN for LSTM input

Train/Test Split

80% training, 20% validation

Model Architecture

```
Embedding Layer → Bidirectional LSTM → Dropout → Bidirectional LSTM → Dense → Dense Softmax
```

Embedding layer maps words to vectors

Bidirectional LSTM reads sequences forward & backward

Dense + Softmax predicts the next word

Training

EarlyStopping monitors val_loss

Best model saved automatically as best_text_generator.h5

Text Generation

Generates text word by word using the trained model

Each next word is predicted using np.argmax on the model output


---

## 💡 Tips to Improve Text Quality

Increase SEQ_LEN to give the LSTM more context

Increase VOCAB_SIZE if dataset is large

Train longer on a GPU for better results

Introduce temperature sampling or top-k/top-p sampling for more varied output


---

## 📈 Example Output
```
Seed: "romeo and juliet"
Generated: "romeo and juliet went to the market and then he said the king will meet"
```

Output depends on dataset and trained model


---


## 📚 Useful References

🤖 TensorFlow LSTM Guide
 – Official guide to using LSTM layers in TensorFlow/Keras.

📝 Keras Tokenizer Documentation
 – Learn how to preprocess text using Keras Tokenizer.


---

## ⚡ License

MIT License – Free to use & modify


---

## 🎯 Author

Christian Jason Ranison
