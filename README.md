# Simple Neural Network from Scratch

This project demonstrates how to build a basic **feedforward neural network** **from scratch**, without using advanced machine learning libraries like **TensorFlow** or **PyTorch**. The goal is to provide a clear and accessible example of how neural networks function at a fundamental level.

---

## **Project Overview**

### 📂 **Project Structure:**
```
neural_network_from_scratch/
|
├─ src/
│   ├─ data/            # Data preparation and input modules
│   │   ├─ data_input_mod.py
│   │   └─ letters_code.py
│   │
│   ├─ model/           # Neural network model and activation functions
│   │   ├─ activation_functions.py
│   │   └─ new_NN.py
│   │
│   ├─ visualization/   # Visualization tools
│   │   └─ data_visualization.py
│   │
│   └─ main.py          # Main script for running the neural network
│
├─ ffnn_from_scratch_jupyter.ipynb         # JupyterNotebook file
|
└─ README.md            # Project description and guide
```

---

## 💡 **Key Concepts:**

1. **Data Representation:**  
   Letters are encoded as **binary vectors**, each represented as a **5x6 grid of pixels** (30 elements).  
<br>
2. **Network Architecture:**  
   - **Input Layer:** 30 neurons (one per pixel).  
   - **Hidden Layer:** 15 neurons (to balance capacity and simplicity).  
   - **Output Layer:** 11 neurons (each representing a letter class).  
<br>
3. **Learning Process:**  
   - **Forward Pass:** Input data flows through the network.  
   - **Loss Calculation:** Using **cross-entropy loss**.  
   - **Backpropagation:** Computes gradients for weight updates.  
   - **Gradient Descent:** Adjusts weights to minimize the loss.  
<br>
4. **Edge Cases:**  
   The network is tested with **noisy inputs** to evaluate its robustness.  

---

## 🛠️ **How to Run:**

1. **Set up the virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Run the main script:**
```bash
python src/main.py
```

3. **Explore the Jupyter Notebook:**  
   You can also run the **Jupyter Notebook** to see a step-by-step guide with explanations:
```bash
jupyter notebook
```

---

## **Visualization:**

- **Classification Results:** Displays the expected vs. predicted letters.  
- **Loss Curve:** Shows how the loss decreases over training epochs.  
 

---

## 🧬 **Diving Deeper:**

- Check the modules in the **`src`** directory to explore the **"raw" algebra** and understand every step of the neural network's training and classification process.
- Each module is **well-documented** with comments and annotations for easy understanding.

---

##  **Next Steps:**

1. **Experiment:** Adjust the number of hidden neurons or the learning rate.  
2. **Add Noise:** Test how robust the model is to distorted inputs.  
3. **Edge Cases:** Provide unusual inputs and observe the behavior.  

---


###  **Contact:**

If you have any questions or suggestions, please reach out via [email](mailto:stanislavkrk@gmail.com).

---

Thank you for exploring this project! I hope it provides valuable insights into building and understanding neural networks **from scratch**.

