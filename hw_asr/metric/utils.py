# Don't forget to support cases when target_text == ''
from typing import List
import editdistance

def calc_cer(target_text: str, predicted_text: str) -> float:
    """
    :param target_text: string
    :param predicted_text: string
    """
    if target_text == '':
        if predicted_text == '':
            return 0
        return 1
    N = len(target_text)
    edit_distance = editdistance.eval(target_text, predicted_text)
    return edit_distance / N

def calc_wer(target_text: str, predicted_text: str) -> float:
    """
    :param target_text: string
    :param predicted_text: string
    """
    if target_text == '':
        if predicted_text == '':
            return 0
        return 1
    target_text = target_text.split(' ')
    predicted_text = predicted_text.split(' ')
    N = len(target_text)
    edit_distance = editdistance.eval(target_text, predicted_text)
    return edit_distance / N