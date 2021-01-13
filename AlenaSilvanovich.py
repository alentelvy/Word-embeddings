import pathlib
from gensim.models import KeyedVectors as kv
import spacy
from scipy.stats import hmean
import json


# chemin vers le fichier des plongement lexicaux
embfile = "/Users/alena/Desktop/syntaxe/frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin"

# Charger les plongements lexicaux en mémoire
wv = kv.load_word2vec_format(embfile, binary=True, encoding='UTF-8', unicode_errors='ignore')

# Charger spacy avec le modèle du français
spacy_nlp = spacy.load('fr_core_news_md')


# Pour chacun des trois aspects, on fournit des mots-exemples qui seront utilisés pour calculer
# des scores de similarité avec chaque token du texte afin de décider s'il exprime un des
# trois aspects
aspects = {
    'nourriture': ['dessert', 'poisson', 'riz', 'pâtes', 'purée', 'viande', 'sandwich', 'frites'],
    'boisson': ['eau', 'vin', 'limonade', 'bière', 'jus', 'thé', 'café'],
    'service': ["service", 'serveur', 'patron', 'employé'],
}


# Similarité moyenne entre un mot et un ensemble de mots : on prend la moyenne harmonique des distances
# puis on la soustrait à 1 pour obtenir la mesure inverse (la similarité), et on arrondit à 4 décimales.
def get_sim(word, other_words):
    if word not in wv.vocab:
        return 0
    dpos = wv.distances(word, other_words)
    d = hmean(abs(dpos))
    return round((1 - d),4)


# Pour un token spacy, cette méthode décide si c'est un terme d'aspect en cherchant l'aspect pour
# lequel il a une similarité maximale (calculée avec les mots-exemples des aspetcs).
# si le score maxi est plus petit que 0.5, il n'y pas d'aspect et la méthode retourne None
def get_aspect_emb(token):
    if token.pos_ != "NOUN":
        return None
    aspect_names = [aspect_name for aspect_name in aspects]
    scores = [(aspect_name,get_sim(token.lemma_, aspects[aspect_name])) for aspect_name in aspect_names]
    scores.sort(key=lambda x:-x[1])
    max_score = scores[0][1]
    max_aspect = scores[0][0] if max_score >= 0.5 else None
    #print(max_aspect)
    return max_aspect



textes = open("/Users/alena/Desktop/syntaxe_git/absa_rest_traindev.txt", "r") 
docs = spacy_nlp.pipe(textes)
resultats = [] 
for doc in docs:
    for sent in doc.sents:
        triplets = []
        for token in sent:
            aspect = get_aspect_emb(token) 
            if aspect is not None:                
                term = token.text
                negation = {'pas': '-', 'non': '-', 'peu': '-', 'guère': '-', 'mal': '-', 'trop': '-'}
#2a. Un adjectif qualificatif qui modifie le terme, il y a une d´ependance de type amod entre le terme et l’expression.
                amod_dep = token.children                                           #get dependencies of term 
                for a in amod_dep:                                                  #loop dependencies 
                    if a.pos_ == 'ADJ':                                             #check if there is adj dependency 
                        if a.head.text == term and a.dep_ == 'amod':                #check if the head of this adj is term and there is amod arc 
                            triplet = []
                            triplet.append(aspect) 
                            triplet.append(term) 
                            a_dep = a.children                                      #get dependencies of this adj 
                            for i in a_dep:                                         #loop dependencies  
                                if i.text in negation.keys() and i.dep_ == 'advmod':#check if the dependent word is in negation dict and there is advmod arc                                    
                                    triplet.append(i.text + '_' + a.text + " NEG_AMOD")
                                    break
                            else: 
                                triplet.append(a.text + " AMOD")
                            triplets.append(triplet)
                            #print('*******AMOD*****', triplet) 
#2b. Un attribut adjectival du sujet dans une clause avec un verbe copule. L’analyse produit une d´ependance nsubj ou nsubj:pass entre l’adjectif et le terme

                if token.head.pos_ == 'ADJ' and (token.dep_ == "nsubj" or token.dep_ == "nsubj:pass"): #check if the head of term is adj and has nsubj dependency
                    triplet = []
                    triplet.append(aspect) 
                    triplet.append(term) 
                    head_dep = token.head.children                                                     #find dependencies of head
                    for j in head_dep:                                                                 #loop dependencies 
                        if j.text in negation.keys() and j.dep_ == 'advmod':                           #if dependent word is in negation dict and has advmod arc
                            triplet.append(j.text + '_' + token.head.text + " NEG_NSUBJ")                    
                            break
                    else: 
                        triplet.append(token.head.text + " NSUBJ")
                    triplets.append(triplet)
                    #print('*******NSUBJ*****', triplet)
                    
#2c. Un adjectif a1 est soit un adjectif qualificatif ou un attribut adjectival du terme: il y a une relation conj entre adj a1 et adj a2                    

                if token.head.pos_ == 'ADJ':                                                        #if term's head is adj 
                    triplet = []
                    triplet.append(aspect) 
                    triplet.append(term) 
                    conj_dep = token.head.children                                                  #get dependencies of the head 
                    for d in conj_dep:                                                              #if in dependencies 
                        if d.pos_ == 'ADJ' and d.dep_ == "conj":                                    #there is an adj with conj arc 
                            d_dep = d.children                                                      #get dependencies of this adj 
                            for c in d_dep:                                                         #loop dependencies 
                                if c.text in negation.keys() and c.dep_ == 'advmod':                #if dependent word is in negation dict and has advmod arc
                                    triplet.append(c.text + '_'+ d.text + " NEG_CONJ") 
                                    break
                            else: 
                                triplet.append(d.text+ " CONJ")
                            triplets.append(triplet)
                            print('*******CONJ*****', triplet)
            #print(triplets)
        if len(triplets) > 0: 
            resultat = {'phrase': sent.text, 'triplets': triplets}
            resultats.append(resultat)                   
            print(resultats)

with open('/Users/alena/Desktop/data.json', 'w', encoding='utf-8') as file:
    json.dump(resultats, file, indent=4)       
                













