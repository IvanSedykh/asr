import editdistance

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    # TODO: your code here

    if target_text == "":
        return 1.0 if predicted_text != "" else 0.0

    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    # TODO: your code here

    if target_text == "":
        return 1.0 if predicted_text != "" else 0.0

    return editdistance.eval(target_text.split(), predicted_text.split()) / len(
        target_text.split()
    )
