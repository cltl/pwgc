import utils


treebank_tagset = {
 'CC',
 'CD',
 'DT',
 'EX',
 'FW',
 'IN',
 'JJ',
 'JJR',
 'JJS',
 'LS',
 'MD',
 'NN',
 'NNP',
 'NNPS',
 'NNS',
 'PDT',
 'POS',
 'PRP',
 'PRP$',
 'RB',
 'RBR',
 'RBS',
 'RP',
 'SYM',
 'TO',
 'Tag',
 'UH',
 'VB',
 'VBD',
 'VBG',
 'VBN',
 'VBP',
 'VBZ',
 'WDT',
 'WP',
 'WP$',
 'WRB'}

wordnet_tagset = {'n', 'v', 'r', 'a'}

pwgc_pos2wordnet = {'NN': 'n', 'VB': 'v', 'JJ': 'a', 'R': 'r', 'J' : 'a'}

universal2wordnet = {'NOUN' : 'n',
                     'VERB' : 'v',
                     'ADJ' : 'a',
                     'ADV' : 'r'}

def treebank2wordnet(treebank_pos):
    """
    a treebank pos tag
    (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

    :param str treebank_pos: treebank pos tag

    :rtype: str
    :return: n, v, a, r, ''
    """

    assert treebank_pos in treebank_tagset, '{treebank_pos} not in treebank tagset'.format_map(locals())

    wordnet_pos = ''
    if treebank_pos.startswith('NN'):
        wordnet_pos = 'n'
    elif treebank_pos.startswith('JJ'):
        wordnet_pos = 'a'
    elif treebank_pos.startswith('VB'):
        wordnet_pos = 'v'
    elif treebank_pos.startswith('RB'):
        wordnet_pos = 'r'

    return wordnet_pos


failure = False
try:
    treebank2wordnet('sdfs')
except AssertionError:
    failure = True
assert failure

assert treebank2wordnet('NNPS') == 'n'
assert treebank2wordnet('JJS') == 'a'
assert treebank2wordnet('RP') == ''



class Token:
    """
    representation of a token

    :
    """
    def __init__(self,
                token_id,
                text,
                lemma,
                lexkeys=set(),
                synsets=set(),
                treebank_pos=None,
                universal_pos=None,
                pos=None):
        self.token_id = token_id
        self.text = text
        self.lemma = lemma
        self.lexkeys = lexkeys
        self.synsets = synsets

        if treebank_pos:

            if treebank_pos in pwgc_pos2wordnet:
                self.pos = pwgc_pos2wordnet[treebank_pos]
            elif treebank_pos in treebank_tagset:
                self.pos = treebank2wordnet(treebank_pos)
            else:
                self.pos = ''

        if universal_pos:
            if universal_pos in universal2wordnet:
                self.pos = universal2wordnet[universal_pos]
            else:
                self.pos = ''



class Sentence:
    """
    representation of sentence

    :ivar list tokens: list of Ctokens instances
    :ivar str id: instance id of sentence
    """
    def __init__(self, id, tokens):
        self.tokens = tokens
        self.id = id


    def sent_in_lstm_format(self, level):
        """
        generate lstm format training examples

        :param str level: sensekey | synset

        see utils.generate_training_instances_v2 for more information

        :rtype: generator
        :return: generator of training examples
        """
        sentence_tokens = []
        sentence_lemmas = []
        sentence_pos = []
        annotations = []


        for token in self.tokens:

            sentence_tokens.append(token.text)
            sentence_lemmas.append(token.lemma)
            sentence_pos.append(token.pos)

            if level == 'sensekey':
                annotations.append(list(token.lexkeys))
            elif level == 'synset':
                annotations.append(list(token.synsets))

        for (target_lemma,
             target_pos,
             token_annotation,
             sentence_tokens,
             training_example,
             target_index) in utils.generate_training_instances_v2(sentence_tokens,
                                                                   sentence_lemmas,
                                                                   sentence_pos,
                                                                   annotations):

            yield training_example


    def sentence(self, instance_id):
        """
        print sentence


        :rtype: str
        :return: the sentence
        """
        tokens = []
        for token_obj in self.token_objs:

            if token_obj.instance_id == instance_id:
                tokens.append('***%s***' % token_obj.token)
            else:
                tokens.append(token_obj.token)

        return ' '.join(tokens)
