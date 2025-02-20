from src.data.letters_code import data_input
from src.data.data_input_mod import (
    letters_list,
    letters_data,
    generate_one_hot_encoding
)
from src.data.user_interaction import (
    choose_letters_dict,
    letter_choose,
)

from src.visualization.data_visualization import data_input_visualization_letters
from src.model.new_NN import (
    random_weights_and_bias,
    training,
    prediction
)


if __name__ == "__main__":
    # --------------------------------------------- Data Preparation ---------------------------------------------------

    data_input = data_input()  # Load the dictionary where keys are letter names and values are their image encodings

    letters_list = letters_list(data_input)  # Retrieve the list of letter names

    x = letters_data(data_input)  # Retrieve the image encodings (binary vectors) as a list

    y_true = generate_one_hot_encoding(letters_list)  # Generate one-hot encodings for the letters' classes

    random = random_weights_and_bias()  # Generate random initial weights for the input-to-hidden layers
    w1 = random[0]
    b1 = random[1]

    random02 = random_weights_and_bias(x=15,
                                        y=11)  # Generate random initial weights for the hidden-to-output layers
    w2 = random02[0]
    b2 = random02[1]

    # ---------------------------------------------- Model Training ----------------------------------------------------

    w1, b1, w2, b2 = training(x, w1, b1, w2, b2, y_true)

    # ---------------------------------------------- Classification ----------------------------------------------------

    choose_letters_dict(letters_list)  # Display available letters

    number = letter_choose()  # Input function: choose which letter index to predict

    letter = x[number]  # This specific letter is passed through the trained network to get probability scores

    predicted_class = prediction(letter, w1, b1, w2, b2, letters_list)  # Run the forward pass and display results

    data_input_visualization_letters(letter,
                                        x[predicted_class])  # Visualize the expected letter vs. the classified one