from __future__ import print_function

import hashlib
import sys

from lxml import html, etree
from xml.sax.saxutils import escape

import utils


class Ctoken:
    def __init__(self, token_id):
        self.token_id = token_id
        self.pos = None
        self.lemma = None
        self.text = None
        self.lexkeys = set()
        self.synsets = set()
        
    def set_pos(self,this_pos):
        self.pos = this_pos
        
    def set_text(self,this_text):
        self.text = this_text
        
    def set_lemma(self, this_lemma):
        self.lemma = this_lemma

    def get_lemma(self):
        return self.lemma
    
    def get_text(self):
        return self.text
    
    def get_pos(self):
        return self.pos
    
    def get_id(self):
        return self.token_id
        
        
    def __str__(self):
        a = '%s %s %s %s' % (self.token_id, str(self.text),str(self.lemma),str(self.pos))
        return a
    
class Csense:
    def __init__(self, lexkey=None,num_sense=None, synset_offset=None):
        self.lexkey = lexkey                #string
        self.num_sense = num_sense          #int
        self.synset_offset = synset_offset  #string
        
    def get_lexkey(self):
        return self.lexkey
    
    def get_num_sense(self):
        return self.num_sense
    
    def get_synset_offset(self):
        return self.synset_offset
    
    def __str__(self):
        s = 'Num_sense: %s\tLexkey: %s\tSynset: %s' % (self.num_sense, self.lexkey, self.synset_offset)
        return s


class Sentence:
    """
    representation of sentence

    :ivar list tokens: list of Ctokens instances
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

    
class Cinstance:
    def __init__(self):
        self.id = ''
        self.docsrc = ''
        self.strategy = ''                  # Which was the strategy to get this instance (expansion type)
        self.lemma = None                   # The lemma of this instance
        self.pos = None                     # The pos of this instance
        self.tokens = []                    # List of Ctoken objects
        self.index_head = []                # Indexes of the head word (could be a multiword) in the self.tokens list
        self.lexkeys = set()                # Gold standard list of sensekeys
        self.sense_rank = None              # Integer corresponding to the ranking of the anotated sense (1, 2, 3...)
        self.confidence_for_senses = {}     # Dictionary assigning a confidence to every sense of the token
        self.is_mfs = False                 # Is a MFS case (at least one of the possible sensekeys is)
        self.is_lfs = False                 # Is a LFS case
        self.annotation_type = None         # manual or auto
        self.cosensekeys = set()            # Co occurent sensekeys in the same synset
        self.mono_cosensekeys = set()       # Subset of cosensekeys that are monosemous (the lemma of the sensekey is monosemous)
        self.cohypo_sensekeys = set()       # Sensekeys in all synsets that are  cohyponyms of the gold synset 
        self.mono_cohypo_sensekeys = set()  # Monosemous sensekeys in all the synsets that are cohyponyms of the gold synset 
         
            
            
    # We compare first the document source and then the id for sorting elememtns
    def __lt__(self,other):
        if self.docsrc < other.docsrc:
            return True
        elif self.docsrc == other.docsrc:
            return self.id < other.id
        else:
            return False
                    
    def set_tokens(self, list_token_objects):
        self.tokens = list_token_objects[:]       
        
    def set_index_head_list(self,list_index_head):
        self.index_head = list_index_head[:]
            
    def get_whole_text(self):
        whole_text = '#'.join([token.text for token in self.tokens]) 
        return whole_text
        
    def get_md5_checksum(self):
        text = '%s_%s_%s_%s' % (self.get_whole_text(), self.get_lemma(), self.get_pos(),str(self.index_head))
        md5_hasher = hashlib.md5()
        md5_hasher.update(text.encode('utf-8'))
        return md5_hasher.hexdigest()     

    def get_lemma(self):
        return self.lemma
    
    def set_lemma(self,lemma):
        self.lemma = lemma
        
    def get_pos(self):
        return self.pos
    
    def set_pos(self,pos):
        self.pos = pos
       
    def set_annotation_type(self, this_type):
        self.annotation_type = this_type
        
    def get_annotation_type(self):
        return self.annotation_type
    
    def set_id(self, this_id):
        self.id = this_id 
    
    def get_id(self):
        return self.id
        
    def set_docsrc(self, this_src):
        self.docsrc = this_src
        
    def set_lexkeys(self, list_skeys):
        self.lexkeys = set(list_skeys)

    def set_sense_rank(self, wn_possible_senses):

        sense_ranks = [wn_possible_senses[lexkey].num_sense for lexkey in self.lexkeys if lexkey in wn_possible_senses]
        if len(sense_ranks) == 0:
            #self.sense_rank = None
            self.sense_rank = 0
        else:
            self.sense_rank = min(sense_ranks)

    def get_lexkeys(self):
        return self.lexkeys
    
    def set_confidence_for_senses(self,this_dict):
        self.confidence_for_senses = this_dict
        
        
    def create_xml_node(self):
        node = etree.Element('instance')
        node.set('id', self.id)
        node.set('docsrc', self.docsrc)
        
        this_string = '<context>'
        start_head = min(self.index_head)
        end_head = max(self.index_head) + 1
        for num_token, token in enumerate(self.tokens):
            if num_token == start_head:
                this_string+=' <head>'+escape(token.get_text())
            elif num_token == end_head:
                this_string+='</head> '+escape(token.get_text())
                it_was_closed = True
            else:
                this_string+=' '+escape(token.get_text())
        
        #If it is the last token of the list, it wont add the </head> tag in the previous loop
        if end_head == len(self.tokens):
            this_string +='</head>'
        this_string+='</context>'
        #print(this_string, end_head, len(self.tokens))
        context_node = etree.fromstring(this_string)
        node.append(context_node)
        return node
    
    
class Clexelt:
    def __init__(self, this_lemma, this_pos):
        self.lemma = this_lemma
        self.pos = this_pos
        self.nltk_wn_pos = None
        self.instances = []
        self.existing_instances = set() #set of md5 checksums for the texts of the instances
        self.wn_possible_senses = {}  #Dictionary from lexkey to Csense
        self._set_nltk_wn_pos()


    def get_possible_senses(self):
        for sense_obj in self.wn_possible_senses.values():
            yield sense_obj
            
    def get_num_senses(self):
        return len(self.wn_possible_senses)
            

    def _set_nltk_wn_pos(self):
        if self.pos[0].lower() == 'n':
            self.nltk_wn_pos = 'n'
        elif self.pos[0].lower() == 'v':
            self.nltk_wn_pos = 'v'
        elif self.pos[0].lower() == 'j':
            self.nltk_wn_pos = 'a'
        elif self.pos[0].lower() == 'r':
            self.nltk_wn_pos = 'r'
            
    def get_nltk_wn_pos(self):
        return self.nltk_wn_pos
            
    def contains_valid_lexkey(self, list_lexkeys):
        contains_valid = False
        for lexkey in list_lexkeys:
            if lexkey in self.wn_possible_senses:
                contains_valid = True
                break
        return contains_valid
            
    def set_wn_possible_skeys(self, wn_reader):
        if wn_reader is not None:
            lemmas = wn_reader.lemmas(self.lemma,pos=self.nltk_wn_pos)
            for int_sense, l in enumerate(lemmas,1):
                #Synset offset is by defualt in, we convert it a 8 character string
                this_synset_offset = str(l.synset().offset())
                num_zeros = 8 - len(this_synset_offset)
                this_synset_offset = '0'*num_zeros + this_synset_offset
                self.wn_possible_senses[l.key()] = Csense(lexkey=l.key(),num_sense=int_sense, synset_offset=this_synset_offset)

    def get_item_key(self):
        return '%s.%s' % (self.lemma,self.pos[0].lower())
        

    
    def add_instance(self,this_instance):
        md5_checksum = this_instance.get_md5_checksum()
        if md5_checksum in self.existing_instances:
            print('Instance with id %s already exists for this lexelt object. Not added' % this_instance.get_id(), file=sys.stderr)           
        else:
            #Add sense rank in case the possible wordnet senses have been set
            this_instance.set_sense_rank(self.wn_possible_senses)
            if len(self.wn_possible_senses) != 0:
                this_instance.set_sense_rank(self.wn_possible_senses)
            
            self.instances.append(this_instance)
            self.existing_instances.add(md5_checksum)
        
    def __repr__(self):
        return self.lemma+' '+self.pos+' '+str(len(self.instances))
    
    def __iter__(self):
        for ins in self.instances:
            yield ins
            
    def create_xml_node(self):
        node = etree.Element('lexelt')
        node.set('item','%s' % self.get_item_key())
        node.set('pos',self.pos)
        for instance in sorted(self.instances):
            node.append(instance.create_xml_node())
        return node
    
    def __len__(self):
        return len(self.instances)
    
    def save_xml_to_file(self,xml_file):
        corpus = etree.Element('corpus')
        corpus.set('lang','en')
        corpus.append(self.create_xml_node())
        train_tree = etree.ElementTree(corpus)
        train_tree.write(xml_file,encoding='UTF-8', pretty_print=True, xml_declaration=True)
        
    def save_key_to_file(self,key_file):
        fd_key = open(key_file,'w')
        this_item = '%s.%s' % (self.lemma, self.pos[0].lower())
        for instance in sorted(self.instances):
            lexkeys = ' '.join(instance.get_lexkeys())
            fd_key.write('%s %s %s\n' % (this_item, instance.get_id(), lexkeys))
        fd_key.close()
