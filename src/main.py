from src.data.letters_code import data_input
from src.data.data_input_mod import letters_list, letters_data, generate_one_hot_encoding
from src.data.user_interaction import choose_letters_dict, letter_choose
from src.visualization.data_visualization import data_input_visualization_letters
from src.model.new_NN import random_weights_and_bias, training, prediction


if __name__ == "__main__":

    # --------------------------------------------- Підготовка даних ---------------------------------------------------

    data_input = data_input() # Завантажуємо словник, який містить ключем назви літер, а значеннями - зображення

    letters_list = letters_list(data_input) # Забираємо список назв літер

    x = letters_data(data_input) # Забираємо списком значення зображень (двійниковий код)

    y_true = generate_one_hot_encoding(letters_list) # Формуємо one-hot список для еталонних передбачень класів літер

    random = random_weights_and_bias() # Генеруємо рандомні стартові ваги для вхідного-прихованого шарів
    w1 = random[0]
    b1 = random[1]

    random02 = random_weights_and_bias(x=15, y=11) # Генеруємо рандомні стартові ваги прихованого-вихідного шарів
    w2 = random02[0]
    b2 = random02[1]

    # ------------------------------------------- Тренування моделі ----------------------------------------------------

    w1, b1, w2, b2 = training(x, w1, b1, w2, b2, y_true)

    # ----------------------------------------------- Класифікація -----------------------------------------------------

    choose_letters_dict(letters_list) # Варіанти літер

    number = letter_choose() # Функція вводу, вибір літери для передбачення

    letter = x[number] # Саме цю літеру пропустить модель на навчених вагах і видасть значення ймовірності

    predicted_class = prediction(letter, w1, b1, w2, b2, letters_list) # Прохід моделі і вивід результатів

    data_input_visualization_letters(letter, x[predicted_class]) # Візуалізація очікування і класифікації