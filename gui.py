import re
import string
import joblib

from pathlib import Path
from nltk.corpus import stopwords
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, END


def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def preprocess_data(text, stop_words):
    text = clean_text(text)
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)

    return text


def check_msg(massage, stop_words, vect, tfidf_transformer, rf, widget):
    preprocess_data(massage, stop_words)

    massages_dtm = vect.transform(massage)
    massages_tfidf = tfidf_transformer.transform(massages_dtm)

    prediction = rf.predict(massages_tfidf)

    if prediction[0] == 0:
        # Изменяем цвет границы на зеленый
        widget.configure(background='#088c25')
    else:
        # Изменяем цвет границы на зеленый
        widget.configure(background='#a30714')


def main():
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/frame0")

    def relative_to_assets(path: str) -> Path:
        return ASSETS_PATH / Path(path)

    window = Tk()

    window.title('NeuroSpam')
    window.geometry("1920x1080")
    window.configure(bg="#010409")

    # logreg = joblib.load('spam_model.pkl')
    rf = joblib.load('first_model.pkl')
    # vect = joblib.load('vect2.pkl')
    vect = joblib.load('first_vect.pkl')
    # tfidf_transformer = joblib.load('first_tfidf_transformer.pkl')
    tfidf_transformer = joblib.load('first_tfidf_transformer.pkl')
    stop_words = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']

    canvas = Canvas(
        window,
        bg="#010409",
        height=1080,
        width=1920,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    canvas.place(x=0, y=0)

    button_image_1 = PhotoImage(
        file=relative_to_assets("button_1.png"))

    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        relief="flat",
        command=lambda: check_msg(
            [entry_1.get('1.0', 'end-1c').strip() if entry_1.get('1.0', 'end-1c').strip() != "" else None], stop_words,
            vect, tfidf_transformer, rf, entry_1),
    )

    button_1.place(
        x=725.0,
        y=765.0,
        width=463.0,
        height=103.0
    )

    image_image_1 = PhotoImage(
        file=relative_to_assets("image_1.png"))
    image_1 = canvas.create_image(
        959.0,
        400.0,
        image=image_image_1
    )

    entry_image_1 = PhotoImage(
        file=relative_to_assets("entry_1.png"))
    entry_bg_1 = canvas.create_image(
        960.0,
        381.5,
        image=entry_image_1
    )
    entry_1 = Text(
        bd=0,
        bg="#161B22",
        fg="#000716",
        highlightthickness=0
    )
    entry_1.place(
        x=280.0,
        y=206.0,
        width=1360.0,
        height=349.0
    )
    entry_1.configure(font=('Roboto', 25), foreground="White", padx=3, pady=18)



    canvas.create_text(
        243.0,
        98.0,
        anchor="nw",
        text="Check the text for spam using modern neural network technologies and ",
        fill="#FFFFFF",
        font=("RobotoRoman Light", 28 * -1)
    )

    canvas.create_text(
        243.0,
        131.0,
        anchor="nw",
        text="protect yourself from unnecessary information.",
        fill="#FFFFFF",
        font=("RobotoRoman Light", 28 * -1)
    )
    window.resizable(True, True)
    window.mainloop()


if __name__ == '__main__':
    main()
