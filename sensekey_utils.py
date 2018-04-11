"""
GOAL:
make it possible to fill the following attributes of Cinstance:

# Co occurent sensekeys in the same synset
1. self.cosensekeys = set()

# Subset of cosensekeys that are monosemous
# (the lemma, pos of the sensekey is monosemous)
2. self.mono_cosensekeys = set()

# Sensekeys in all synsets that are  cohyponyms of the gold synset
3. self.cohypo_sensekeys = set()

# Monosemous sensekeys in all the synsets that are cohyponyms of the gold synset
4. self.mono_cohypo_sensekeys = set()


function to expand a set of lemma, pos combinations to all relevant lemma, pos
combinations given this seed set with respect to attributes described in
1. 2. 3. and 4.
"""
from nltk.corpus import wordnet as wn



def get_lemma_pos_of_sensekey(sense_key):
    """
    lemma and pos are determined for a wordnet sense key

    >>> get_lemma_pos_of_sensekey('life%1:09:00::')
    ('life', 'n')

    :param str sense_key: wordnet sense key

    :rtype: tuple
    :return: (lemma, n | v | r | a | u)
    """
    if '%' not in sense_key:
        return '', 'u'

    lemma, information = sense_key.split('%')
    int_pos = information[0]

    if int_pos == '1':
        this_pos = 'n'
    elif int_pos == '2':
        this_pos = 'v'
    elif int_pos in {'3', '5'}:
        this_pos = 'a'
    elif int_pos == '4':
        this_pos = 'r'
    else:
        this_pos = 'u'

    return lemma, this_pos


def load_mapping_sensekey2offset(path_to_index_sense, wn_version):
    """
    extract mapping sensekey2offset from index.sense file

    :param str path_to_index_sense: path to wordnet index.sense file
    :param str wn_version: 171 | 21 | 30

    :rtype: dict
    :return: mapping sensekey (str) -> eng-WN_VERSION-OFFSET-POS
    """
    sensekey2offset = dict()

    with open(path_to_index_sense) as infile:
        for line in infile:
            sensekey, synset_offset, sense_number, tag_cnt = line.strip().split()

            lemma, pos = get_lemma_pos_of_sensekey(sensekey)

            assert pos != 'u'
            assert len(synset_offset) == 8

            identifier = 'eng-{wn_version}-{synset_offset}-{pos}'.format_map(locals())

            sensekey2offset[sensekey] = identifier

    return sensekey2offset

def get_lemma_pos_of_sensekey(sense_key):
    """
    lemma and pos are determined for a wordnet sense key

    >>> get_lemma_pos_of_sensekey('life%1:09:00::')
    ('life', 'n')

    :param str sense_key: wordnet sense key

    :rtype: tuple
    :return: (lemma, n | v | r | a | u)
    """
    if '%' not in sense_key:
        return '', 'u'

    lemma, information = sense_key.split('%')
    int_pos = information[0]

    if int_pos == '1':
        this_pos = 'n'
    elif int_pos == '2':
        this_pos = 'v'
    elif int_pos in {'3', '5'}:
        this_pos = 'a'
    elif int_pos == '4':
        this_pos = 'r'
    else:
        this_pos = 'u'

    return lemma, this_pos


def get_monosemous_sensekeys(wn_instance, set_of_sensekeys):
    """
    return monosemous sensekeys, which means that the lemma of the
    sensekey in the pos of the sensekey is monosemous in wordnet

    >>> get_monosemous_sensekeys(wn, {'poland%1:15:00::', 'poky%5:00:00:slow:01'})
    {'poland%1:15:00::'}

    :param nltk.corpus.reader.wordnet.WordNetCorpusReader wn_instance:
    instance of wordnet in nltk
    :param set set_of_sensekeys: set of wordnet sensekeys

    :rtype: set
    :return: set of wordnet sensekeys of which the lemma and pos
    of the sensekeys are monosemous in wordnet
    """
    mon_sensekeys = set()

    for sensekey in set_of_sensekeys:
        lemma, pos = get_lemma_pos_of_sensekey(sensekey)
        synsets = wn_instance.synsets(lemma, pos=pos)
        if len(synsets) == 1:
            mon_sensekeys.add(sensekey)

    return mon_sensekeys


def get_co_sensekeys(synset, main_sensekey=None):
    """
    obtain the sensekeys that occur in the same synset as the main_sensekey

    >>> get_co_sensekeys(wn.synset('cat.n.1'), main_sensekey='cat%1:05:00::')
    {'true_cat%1:05:00::'}

    :param nltk.corpus.reader.wordnet.Synset synset: synset wordnet instance
    as defined in the nltk
    :param str main_sensekey: a wordnet sensekey (e.g. cat%1:05:00::)

    :rtype: set
    :return: set of all sensekeys that are in the same synset of the
    main_sensekey
    """
    co_sensekeys = set()

    for lemma in synset.lemmas():
        co_sensekey = lemma.key()
        if co_sensekey != main_sensekey:
            co_sensekeys.add(co_sensekey)

    return co_sensekeys


def get_cohypo_sensekeys(synset):
    """
    obtain all sensekeys from synsets that are cohyponyms of the main synset

    >>> sorted(get_cohypo_sensekeys(wn.synset('cat.n.1')))
    ['big_cat%1:05:00::', 'cat%1:05:02::']

    :param nltk.corpus.reader.wordnet.Synset synset: synset wordnet instance
    as defined in the nltk

    :rtype: set
    :return: set of sensekeys from synsets that are cohyponyms from the
    param synset
    """
    synset_sensekeys = {lemma.key() for lemma in synset.lemmas()}
    co_hypo_sensekeys = set()

    for hypernym in synset.hypernyms():
        for co_hyponym in hypernym.hyponyms():

            if co_hyponym != synset:
                sensekeys = {lemma.key()
                             for lemma in co_hyponym.lemmas()}
                for sensekey in sensekeys:
                    if sensekey not in synset_sensekeys:
                        co_hypo_sensekeys.add(sensekey)

    return co_hypo_sensekeys


def hyponym_sensekeys(synset, depth_synset=None, max_depth_from_synset=None):
    """
    given a synset, this function returns all sensekeys from all hyponyms
    in the tree of the synset.

    >>> 'domestic_cat%1:05:00::' in hyponym_sensekeys(wn.synset('cat.n.1'), max_depth_from_synset=1)
    True

    >>> hyponym_sensekeys(wn.synset('entity.n.1'), depth_synset=5)
    set()

    >>> len(hyponym_sensekeys(wn.synset('cat.n.1')))
    87

    :param nltk.corpus.reader.wordnet.Synset synset: synset wordnet instance
    as defined in the nltk
    :param range depth_synset: range of depths of synsets that from which
    you want the hyponym sensekeys

    :param  int max_depth_from_synset: how deep do you want to go to obtain
    sensekeys from hyponyms

    :rtype: set
    :return: set of hyponyms sensekeys
    """
    sensekeys = set()

    # check synset depth
    if depth_synset:
        if synset.min_depth() < depth_synset:
            return sensekeys

    hyponyms = synset.hyponyms()
    cur_depth = 1
    while hyponyms:

        if max_depth_from_synset:
            if cur_depth > max_depth_from_synset:
                break

        keys = {lemma.key()
                for hyponym in hyponyms
                for lemma in hyponym.lemmas()}
        sensekeys.update(keys)

        hyponyms = [new_hyponym
                    for hyponym in hyponyms
                    for new_hyponym in hyponym.hyponyms()]
        cur_depth += 1

    return sensekeys

def expand_lemma_pos(wn_instance, set_of_lemma_pos, exclude_monosemous=False):
    """
    for every sense of a lemma, pos, the seed set is expanded with:
    1. the lemma, pos of the co_sensekeys
    2. the lemma, pos of the sensekeys in the cohyponym synsets

    :param nltk.corpus.reader.wordnet.WordNetCorpusReader wn_instance:
    instance of wordnet in nltk
    :param set set_of_lemma_pos: set of tuples (lemma, pos)
    :rtype: set
    :return: set of lemma pos combinations
    """
    all_sensekeys = set()
    expanded_lemma_pos = set_of_lemma_pos.copy()

    for lemma, pos in set_of_lemma_pos:

        for synset in wn_instance.synsets(lemma, pos=pos):
            sensekeys = {lemma.key() for lemma in synset.lemmas()}
            co_sensekeys = get_co_sensekeys(synset)
            # co_hypo_sensekeys = get_cohypo_sensekeys(synset)

            for keys in [sensekeys, co_sensekeys]:
                if exclude_monosemous:
                    monosemous = get_monosemous_sensekeys(wn_instance,
                                                          keys)
                    keys = keys - monosemous
                all_sensekeys.update(keys)

    for sensekey in all_sensekeys:
        lemma, pos = get_lemma_pos_of_sensekey(sensekey)
        expanded_lemma_pos.add((lemma, pos))

    return expanded_lemma_pos




def add_sense_info_to_clexelt(c_lexelt_obj,
                              wn_instance,
                              debug=False):
    """
    the following attributes have to have a value in each Cinstance instance:
    (set under c_lexelt_instance.existing_instances)
    1. lemma
    2. nltk_wn_pos
    3. lexkeys

    the following attributes are updated for each object in
    c_lexelt_instance.instances

    # Co occurent sensekeys in the same synset
    1. cosensekeys

    # Subset of cosensekeys that are monosemous
    # (the lemma of the sensekey is monosemous)
    2. mono_cosensekeys

    # Sensekeys in all synsets that are  cohyponyms of the gold synset
    3. cohypo_sensekeys

    # Monosemous sensekeys in all the synsets
    # that are cohyponyms of the gold synset
    4. mono_cohypo_sensekeys

    :param my_data_classes.Clexelt c_lexelt_obj: class instance defined in
    my_data_classes.Clexelt representing a lemma, pos combination
    with all training instance for this lemma, pos

    :param nltk.corpus.reader.wordnet.WordNetCorpusReader wn_instance:
    instance of wordnet in nltk

    :param bool debug: if set to True, debug mode is on

    :rtype: my_data_classes.Clexelt
    :return: my_data_classes.Clexelt with four attributes updated
    for each training instance
    """
    gold_wn_pos = c_lexelt_obj.get_nltk_wn_pos()
 
    for c_instance in c_lexelt_obj.instances:

        gold_keys = c_instance.lexkeys

        keys_synsets = {}
        for sensekey, c_sense_obj in c_lexelt_obj.wn_possible_senses.items():
            if sensekey in gold_keys:
                offset = c_sense_obj.synset_offset
                synset = wn_instance._synset_from_pos_and_offset(gold_wn_pos,
                                                            int(offset))
                keys_synsets[sensekey] = synset


        all_co_sensekeys = {co_sensekey
                    for gold_key, gold_synset in keys_synsets.items()
                    for co_sensekey in get_co_sensekeys(gold_synset,
                                                        main_sensekey=gold_key)}
        mon_co_sensekeys = get_monosemous_sensekeys(wn_instance,
                                                    all_co_sensekeys)

        all_co_hypo_sensekeys = {co_hypo_sensekey
                            for gold_key, gold_synset in keys_synsets.items()
                            for co_hypo_sensekey in
                            get_cohypo_sensekeys(gold_synset)}

        mon_co_hypo_sensekeys = get_monosemous_sensekeys(wn_instance,
                                                         all_co_hypo_sensekeys)

        # update attributes
        c_instance.cosensekeys = all_co_sensekeys
        c_instance.mono_cosensekeys = mon_co_sensekeys
        c_instance.cohypo_sensekeys = all_co_hypo_sensekeys
        c_instance.mono_cohypo_sensekeys = mon_co_hypo_sensekeys

        if debug:
            print()
            print('lemma: ', c_instance.lemma)
            print('nltk_pos', c_lexelt_obj.get_nltk_wn_pos())
            print('gold keys', c_instance.lexkeys)
            print('cosensekeys', c_instance.cosensekeys)
            print('mon cosensekeys', c_instance.mono_cosensekeys)
            print('co_hypo_sensekeys', c_instance.cohypo_sensekeys)
            print('mon_co_hypo_sensekeys', c_instance.mono_cohypo_sensekeys)
            input('continue?')

    return c_lexelt_obj


if __name__ == '__main__':
    # TODO: test function add_sense_info_to_clexelt
    a_wn_instance = wn
    a_c_lexelt_instance = None
    updated_c_lexelt_instance = add_sense_info_to_clexelt(a_c_lexelt_instance,
                                                         a_wn_instance,
                                                         debug=True)
