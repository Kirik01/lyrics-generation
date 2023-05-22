# Telegram lyrics generator 
A project of making a Telegram chat service on the basis of a neural network 
that was trained on Leps lyrics to produce 'predictions' in accord with the questions asked

Original concept is borrowed from: https://github.com/dlebech/lyrics-generator/

Actions taken and features introduced in this project:
 - Preparation of feed for a model on the basis of Leps songs, fetched from https://text-you.ru/
 - Training the model on the resulting dataset
 - Embedding the lyrics-generator script into the Telegram 'echo bot'
 - Deployment of the service in YandexCloud

# Telegram bot
The bot is called `Leps_generator`.
The bot lives here: https://t.me/Liam_Machina_bot

# Feed preparation
`song_texts_parsing.ipynb` contains commands to fetch, parse, normalize Leps lyrics and make a feed for the model. 
Herewith file `leps_text.csv` is the resulting feed, whereas `export/model.h5` is a model, respectively trained on its basis.     

# Embeddings
- Download `glove.6B.50d.txt` file from:http://nlp.stanford.edu/data/glove.6B.zipin and put it in the subdirectory `data/`.
- Or create your own embedding: make `songdata.csv` file from above and then run:
```shell
python -m lyrics.embedding --name-suffix _myembedding
```
- This will create `word2vec_myembedding.model` and `word2vec_myembedding.txt` files in the default subdirectory `data/`.

# Training the model
To train your own model run one of the following:
```shell
python -m lyrics.train --early-stopping-patience 50 --artists '*'
python -m lyrics.train --embedding-file ./word2vec_myembedding.txt
```
Check `python -m lyrics.train -h` for other options
Check `lyrics/config.py` for training configuration 

# Run main()
To make it work you will have to specify your own telegram token in tg_token

# Install dependencies
Requires Python 3.7+.
```shell
pip install -r requirements.txt
```
