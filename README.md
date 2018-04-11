# Script to load Princeton Wordnet Gloss Corpus into python classes

```
bash install.sh
python pwhc_to_ulm.py -i pwgc -o ulm
python convert_ulm_to_lstm_format.py
```

In the folder **ulm** the corpus is represented as **my_data_classes.Cinstance** objects.

In **output/instances.txt** the corpus is represented as:

```
the first Pope born in Poland ; the first Pope not---eng-30-00024073-r born in Italy in 450 years
```