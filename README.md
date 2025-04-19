# Recipe Generation 

This project is a recipe generation chatbot built using TensorFlow and Keras. It trains a Recurrent Neural Network (RNN) to generate recipes based on a dataset of raw recipe JSON files.
It generates recipe titles, ingredients, and instructions.

## Features

- **Recipe Generation:** Generates new recipes based on a starting title using LSTM-based RNN architecture.
- **Dataset Validation:** Filters and validates the dataset to ensure all recipes contain required fields (title, ingredients, and instructions).
- **Tokenizer:** Tokenizes text data for character-level processing.
- **Model Training:** Trains an RNN model with a checkpointing mechanism to ensure progress is saved.
- **Text Generation:** Generates realistic recipe texts based on a given start string.

## Setup

### Prerequisites

To run this project, make sure you have the following installed:

- Python 3.x
- TensorFlow
- Matplotlib
- NumPy

You can install the necessary dependencies with the following command:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file can include:

```
tensorflow
matplotlib
numpy
```

### Dataset Extraction

The script expects a zip file (`recipes_raw.zip` you can find it here `https://eightportions.com/`) containing the recipe JSON files. Here's how the dataset is handled:

1. **Directory Setup:**  
   The script first checks if a directory called `tmp` exists in the current working directory. If it doesn't, it creates one using the following path:  
   `./tmp`.

2. **Dataset Extraction:**  
   The script looks for the `recipes_raw.zip` file either in the current directory or in the `tmp` directory. If the zip file is found, it is extracted into the `./tmp` directory. After extraction, the dataset is available in the path:  
   `./tmp/datasets/`.  
   This folder will contain the necessary JSON files (e.g., `recipes_raw_nosource_ar.json`, `recipes_raw_nosource_epi.json`, `recipes_raw_nosource_fn.json`).

3. **Dataset Validation:**  
   Once extracted, the dataset is processed, validated, and formatted into a string representation, which is used for training the model.

### Running the Script

Once the dataset is extracted and the required packages are installed, you can run the main script:

```bash
python app.py
```

The script will:

1. Load and validate the dataset.
2. Tokenize the dataset.
3. Build and train a TensorFlow RNN model using the processed dataset.
4. Save the model weights during training.
5. Generate a recipe using a pre-trained model.

### Example Output

After training, the script will output a generated recipe, like so:

```
✨ Generated Recipe:
📗 Paneer Butter Masala
🥕
• Paneer
• Tomatoes
• Butter
• Onion
📝
▪︎ Heat butter in a pan.
▪︎ Add chopped onions and cook until golden brown.
▪︎ Add chopped tomatoes and cook until soft.
▪︎ Add paneer cubes and cook for 10 minutes.
```

### Customization

You can adjust the following settings:

- **Dataset location:** Modify the path to your dataset or replace the zip file in the appropriate directory.
- **Hyperparameters:** The model's hyperparameters (embedding dimension, RNN units, batch size, etc.) can be adjusted in the script.

### Checkpoints

The model saves checkpoints during training in the `./training_checkpoints` directory. You can use these checkpoints to continue training or to load a pre-trained model for text generation.

### Error Handling

- If the `recipes_raw.zip` file is not found, the script will display:  
  `❌ Dataset not found!`
