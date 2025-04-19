import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import platform
import time
import pathlib
import os
import json
import zipfile

# Constants
DEBUG = True
DEBUG_EXAMPLES = 10
STOP_WORD_TITLE = '📗 '
STOP_WORD_INGREDIENTS = '\n🥕\n\n'
STOP_WORD_INSTRUCTIONS = '\n📝\n\n'
STOP_SIGN = '<STOP>'  # Changed from ␣ to ASCII-safe
MAX_RECIPE_LENGTH = 2000


def load_dataset(cache_dir, silent=False):
    dataset_file_names = [
        'recipes_raw_nosource_ar.json',
        'recipes_raw_nosource_epi.json',
        'recipes_raw_nosource_fn.json',
    ]
    dataset = []
    for dataset_file_name in dataset_file_names:
        dataset_file_path = f'{cache_dir}/datasets/{dataset_file_name}'
        with open(dataset_file_path) as dataset_file:
            json_data_dict = json.load(dataset_file)
            json_data_list = list(json_data_dict.values())
            dataset += json_data_list
    return dataset


def recipe_validate_required_fields(recipe):
    required_keys = ['title', 'ingredients', 'instructions']
    if not recipe:
        return False
    for required_key in required_keys:
        if not recipe.get(required_key):
            return False
        if isinstance(recipe[required_key], list) and len(recipe[required_key]) == 0:
            return False
    return True


def recipe_to_string(recipe):
    noize_string = 'ADVERTISEMENT'
    title = recipe['title']
    ingredients = recipe['ingredients']
    instructions = recipe['instructions'].split('\n')

    ingredients_string = ''
    for ingredient in ingredients:
        ingredient = ingredient.replace(noize_string, '')
        if ingredient.strip():
            ingredients_string += f'• {ingredient.strip()}\n'

    instructions_string = ''
    for instruction in instructions:
        instruction = instruction.replace(noize_string, '')
        if instruction.strip():
            instructions_string += f'▪︎ {instruction.strip()}\n'

    return f'{STOP_WORD_TITLE}{title}\n{STOP_WORD_INGREDIENTS}{ingredients_string}{STOP_WORD_INSTRUCTIONS}{instructions_string}'


def recipe_sequence_to_string(sequence):
    return tokenizer.sequences_to_texts([sequence])[0]


def split_input_target(recipe):
    return recipe[:-1], recipe[1:]


def build_model_1(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim
    ))
    model.add(tf.keras.layers.LSTM(
        units=rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform'
    ))
    model.add(tf.keras.layers.Dense(vocab_size))
    return model


def build_model_for_generation(vocab_size, embedding_dim, rnn_units):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(batch_shape=(1, None)))
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))
    model.add(tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True))
    model.add(tf.keras.layers.Dense(vocab_size))
    return model


def generate_text(model, start_string, gen_length=500, temperature=1.0):
    input_eval = tokenizer.texts_to_sequences([start_string])[0]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    model.layers[1].reset_states()

    for _ in range(gen_length):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(tokenizer.index_word.get(predicted_id, ''))

        if tokenizer.index_word.get(predicted_id, '') == STOP_SIGN:
            break

    return start_string + ''.join(text_generated)


def main():
    # Setup
    cache_dir = './tmp'
    pathlib.Path(cache_dir).mkdir(exist_ok=True)

    # Load dataset
    zip_path = os.path.join(cache_dir, 'recipes_raw.zip')
    if os.path.exists('recipes_raw.zip'):
        print("Using local zip from current directory")
        with zipfile.ZipFile('recipes_raw.zip', 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
    elif os.path.exists(zip_path):
        print("Using zip from cache directory")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
    else:
        print("❌ Dataset not found!")
        return

    dataset_raw = load_dataset(cache_dir)
    dataset_validated = [r for r in dataset_raw if recipe_validate_required_fields(r)]
    dataset_stringified = [recipe_to_string(r) + STOP_SIGN for r in dataset_validated]

    print('Dataset size BEFORE validation:', len(dataset_raw))
    print('Dataset size AFTER validation:', len(dataset_validated))
    print('✅ Processed:', len(dataset_stringified))

    if DEBUG:
        for i, r in enumerate(dataset_stringified[:DEBUG_EXAMPLES]):
            print(f"\nRecipe #{i+1}\n{'-'*30}\n{r}")

    # Tokenizer
    global tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        char_level=True,
        filters='',
        lower=False,
        split=''
    )
    tokenizer.fit_on_texts([STOP_SIGN])
    tokenizer.fit_on_texts(dataset_stringified)
    tokenizer.index_word = {v: k for k, v in tokenizer.word_index.items()}  # ✅ Required for generation
    VOCABULARY_SIZE = len(tokenizer.word_index) + 1

    print("\nTokenizer vocab size:", VOCABULARY_SIZE)

    # Vectorize
    dataset_sequences = tokenizer.texts_to_sequences(dataset_stringified)
    dataset_vectorized_padded = tf.keras.preprocessing.sequence.pad_sequences(
        dataset_sequences,
        padding='post',
        truncating='post',
        maxlen=MAX_RECIPE_LENGTH
    )

    # TF Dataset
    dataset = tf.data.Dataset.from_tensor_slices(dataset_vectorized_padded)
    dataset_targeted = dataset.map(split_input_target)

    BATCH_SIZE = 8
    SHUFFLE_BUFFER_SIZE = 1000
    dataset_train = dataset_targeted.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()

    # Build model
    EMBEDDING_DIM = 256
    RNN_UNITS = 512
    model_1 = build_model_1(VOCABULARY_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
    model_1.summary()

    # Checkpoints
    checkpoint_dir = './training_checkpoints'
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5"),
        save_weights_only=True
    )

    # Compile & Train
    model_1.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model_1.fit(dataset_train, epochs=20, steps_per_epoch=20, callbacks=[checkpoint_callback])

    # Generate
    print("\n🔄 Generating recipe...")

    gen_model = build_model_for_generation(VOCABULARY_SIZE, EMBEDDING_DIM, RNN_UNITS)

    latest = tf.train.get_checkpoint_state(checkpoint_dir)
    if latest and latest.all_model_checkpoint_paths:
        last_ckpt = latest.all_model_checkpoint_paths[-1] + ".weights.h5"
    else:
        last_ckpt = os.path.join(checkpoint_dir, "ckpt_5.weights.h5")

    print("Loading weights from:", last_ckpt)
    gen_model.load_weights(last_ckpt)
    gen_model.build(tf.TensorShape([1, None]))

    generated = generate_text(gen_model, start_string=STOP_WORD_TITLE + "Paneer Butter Masala")
    print(f"\n✨ Generated Recipe:\n{generated}")


# Run
if __name__ == "__main__":
    main()
