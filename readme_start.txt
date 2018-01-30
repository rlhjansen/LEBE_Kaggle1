Maak een directory die heet: split_classes in de map boven de git
 > python split_classes.py
     - Memory error? Ga naar BATCH_SIZE (regel 16) en maak die kleiner. Reset
       voordat je het programma opnieuw runt.
     - RESET: Verwijder altijd de inhoud van de split_classes 
       directory als je het programma opnieuw runt.
    
 > python delta_word.py
     - Memory error? Ga naar BATCH_SIZE (regel 12) en maak die kleiner, reset
       vervolgens.
     - De waarde VEC_LEN zullen we (wellicht) veranderen, die staat op regel 14
     - RESET: Voor de reset is het belangrijk tijdens welk proces je het
       programma hebt gestopt.
        - tijdens counting: Je hoeft niks te doen
        - tijdens comparing: Je hoeft niks te doen
        - tijdens mapping: ga naar main() op regel 446 en vul geef het de 
            variabele PATH_OUT mee, zodat je de vector in kan laden i.p.v. 
            opnieuw generen. Moet je generen? Verwijder dan 
            'delta_vec_[VEC_LEN].tsv' in de directory boven de git.
        - tijdens preprocessing: preprocessing wordt gekenmerd door de prints:
            "Now loading batch: x..."
            "...Extracting class values..."
            "...Converting to input..."
            "...Storing"
            Voor inladen:
                ga naar main() op regel 446 en zorg dat hij de volgende 3
                globale variabelen krijgt megegeven:
                PATH_OUT, PATH_DMAP, PATH_LMAP
                Verwijder daarnaast de files:
                 - 'delta_train_[VEC_LEN].tsv'
                 - 'delta_val_[VEC_LEN].tsv'
            Voor volledige reset:
                Verwijder de volgende files:
                 - 'delta_vec_[VEC_LEN].tsv'
                 - 'delta_occ_[VEC_LEN].tsv'
                 - 'delta_labels_[VEC_LEN].tsv'
                 - 'delta_train_[VEC_LEN].tsv'
                 - 'delta_val_[VEC_LEN].tsv'
                En zorg er voor dat main() op regel 446 geen parameters krijgt
                meegegeven.

 > python train_val_plot.py
     - Eerste run: Zorg er voor dat main() (rond regel 179) alleen de parameter
       TRAINLAYERS krijgt meegegeven.
     - Niet eerste run: Zorg er voor dat main() (rond regel 179) de parameters
       TRAINLAYERS, SAVED_NN, False   krijgt meegegeven
     - Verander de BATCH_SIZE niet.
     - Heb je de waarde VEC_LEN veranderd in delta_word.py? Pas dan ook deze 
       waarde aan in deze file (VEC_LEN, regel 12)
     - Pas eventueel LAYERS aan (regel 35)
     - RESET: verwijder of verplaats de files in de folder boven de git:
         - 'delta_train_tst_[VEC_LEN].tsv'
         - 'NN_pickle_[LAYERS]_layer.pkl'
         - 'train_error_values.txt'
         - 'val_error_values.txt'
         - 'x_values_written.txt'
        