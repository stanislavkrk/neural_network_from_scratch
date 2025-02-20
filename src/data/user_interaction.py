"""
Module: user_interaction

This module contains functions that realise user input and output,
including:
- Displaying available letters for recognition (choose_letters_dict).
- Prompting the user to select a letter index (letter_choose).
"""

def choose_letters_dict(letters_list):
    """
    Displays the list of available letters for recognition.

    :param letters_list: A list of letter names.
    :return: None
    """
    letters_dict = {name: value for name, value in zip(range(0,11), letters_list)}
    for name, value in letters_dict.items():
        print(name, value)

def letter_choose():
    """
    Function to select a letter from the list for recognition.

    :return: The index (integer) of the chosen letter.
    """
    number_list = list(range(0,11))
    while True:
        number = int(input('Enter the letters number for recognition: '))
        if number in number_list:
            return number
        else:
            print('Incorrect number, please try again.')