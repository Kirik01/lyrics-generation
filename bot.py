#############################

"""Predict from a previously generated song model."""
import argparse
import json
import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

from lyrics import util


model = 'export/2023-05-17T192425/model.h5'
tokenizer = 'export/2023-05-17T192425/tokenizer.pickle'


def load_model(model_filename):
    return tf.keras.models.load_model(
        model_filename, custom_objects={"KerasLayer": hub.KerasLayer}
    )


def softmax_sampling(probabilities, randomness, seed=None):
    """Returns the index of the highest value from a softmax vector,
    with a bit of randomness based on the probabilities returned.

    """
    if seed:
        np.random.seed(seed)
    if randomness == 0:
        return np.argmax(probabilities)
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(probabilities) / randomness
    exp_probabilities = np.exp(probabilities)
    probabilities = exp_probabilities / np.sum(exp_probabilities)
    return np.argmax(np.random.multinomial(1, probabilities, 1))



def generate_lyrics(model, tokenizer, text_seed, song_length, randomness=0, seed=None):
    """Generate a new lyrics based on the given model, tokenizer, etc.

    Returns the final output as both a vector and a string.

    """
    # The sequence length is the second dimension of the input shape. If the
    # input shape is (None,), the model uses the transformer network which
    # takes a string as input!
    input_shape = model.inputs[0].shape
    seq_length = -1
    if len(input_shape) >= 2:
        print("Using integer sequences")
        seq_length = int(input_shape[1])
    else:
        print("Using string sequences")

    # Create a reverse lookup index for integers to words
    rev = {v: k for k, v in tokenizer.word_index.items()}

    spacer = "" if tokenizer.char_level else " "

    text_output = tokenizer.texts_to_sequences([text_seed])[0]
    text_output_str = spacer.join(rev.get(word) for word in text_output)
    while len(text_output) < song_length:
        if seq_length != -1:
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                [text_output], maxlen=seq_length, padding="post"
            )
        else:
            padded = np.array([text_output_str])
        next_word = model.predict_on_batch(padded)
        next_word = softmax_sampling(next_word[0], randomness, seed=seed)
        text_output.append(next_word)
        text_output_str += f"{spacer}{rev.get(next_word)}"
    return text_output, text_output_str




def lyrics(text, length, model, tokenizer):
    model = load_model(model)

    tokenizer = util.load_tokenizer(tokenizer)

    print(f'Generating lyrics from "{text}"...')
    seed = (np.random.randint(np.iinfo(np.int32).max)
    )

    raw, text = generate_lyrics(
        model, tokenizer, text, length, 0.0, seed=seed
    )
    # print(text)
    # print()
    # print(f"Random seed (for reproducibility): {seed}")
    return text


# length = 20
# text = 'как дела?'
# predict_text = lyrics(text, length, model, tokenizer)
# print('Prediction:', predict_text)


########################
import logging

from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

tg_token = '6175185809:AAHMRh-TFJdW3n0V0Cdcp3WLwbJlsnhb_K0'


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def predict_lyrics_tg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    length = 20
    test_answer = lyrics(update.message.text, length, model, tokenizer)
    await update.message.reply_text(test_answer)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(tg_token).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, predict_lyrics_tg))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
