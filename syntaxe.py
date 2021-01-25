import pathlib
from gensim.models import KeyedVectors as kv
import spacy
from scipy.stats import hmean
import json
import sys


# chemin vers le fichier des plongement lexicaux
embfile = "./frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin"


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
    return max_aspect

argument = sys.argv[1]
with open (argument, "r", encoding="utf-8") as f:
    textes = f.readlines()
    docs = spacy_nlp.pipe(textes)
    resultats = [] 
    for doc in docs:
        for sent in doc.sents:
            triplets = []
            for token in sent:
                aspect = get_aspect_emb(token) # NOUN déjà vérifié dans cette fonction, donc pas de verification de token.pos_ == "NOUN" ci-dessous
                if aspect is not None:                
                    term = token.text
                    negation = {'pas': '-', 'non': '-', 'peu': '-', 'guère': '-', 'mal': '-', 'trop': '-'} #dict avec modifieurs adverbials négatifs comme clé

                    #2a. chercher un adjectif qualificatif qui modifie le terme. Dépendence AMOD entre le terme et l’expression.
                    amod_dep = token.children                                               
                    for a in amod_dep:                                                      
                        if a.pos_ == 'ADJ' and a.dep_ == 'amod':                                                                                              
                            triplet = []
                            neg_amod = False
                            neg_conj = False                                                    
                            a_dep = a.children                                              
                            #2d. chercher un modifieur adverbial négatif pour adjectif qualificatif
                            for achild in a_dep:
                                if achild is not None: 
                                    if achild.dep_ == 'advmod':                                                
                                        if achild.text in negation.keys():         
                                            neg_amod = True                                   
                                        else: 
                                            neg_amod = False
                                    #2c. chercher un adjectif coordonné pour adjectif qualificatif
                                    if achild.pos_ == 'ADJ' and achild.dep_ == "conj": 
                                        if achild.head.head.text == term :
                                            achilddep = achild.children                                                                                   
                                            for cchild in achilddep:
                                                if cchild is not None:
                                                    #2d. chercher un modifieur adverbial négatif pour adjectif coordonné
                                                    if cchild.dep_ == 'advmod':
                                                        if cchild.text in negation.keys(): 
                                                            neg_conj = True                                                
                                                        else: 
                                                            neg_conj = False  

                                            #écrire un adjectif coordonné et son modifieur adverbial négatif 
                                            if neg_conj == True: 
                                                triplet = [aspect, term, cchild.text + '_' + achild.text] #NEG_CONJ
                                                triplets.append(triplet)
                                            else: 
                                                triplet = [aspect, term,  achild.text] #CONJ
                                                triplets.append(triplet) 

                            #écrire un adjectif qualificatif et son modifieur adverbial négatif 
                            if neg_amod == True: 
                                triplet = [aspect, term,  achild.text + '_' + a.text] #NEG_AMOD
                                triplets.append(triplet)
                            else: 
                                triplet = [aspect, term,  a.text] #AMOD
                                triplets.append(triplet)
                           
                    #2b. un attribut adjectival du sujet dans une clause avec un verbe copule. Une dépendance nsubj ou nsubj:pass entre l’adjectif et le terme
                    if token.head.pos_ == 'ADJ' and (token.dep_ == "nsubj" or token.dep_ == "nsubj:pass"):   
                        triplet = []                   
                        head_dep = token.head.children                                                      
                        for headchild in head_dep: 
                            if headchild is not None:                                                        
                                neg_nsubj = False
                                #2d. chercher un modifieur adverbial négatif pour un attribut adjectival                                                         
                                if headchild.dep_ == 'advmod':
                                    if headchild.text in negation.keys():                           
                                        neg_nsubj = True
                                    else: 
                                        neg_nsubj = False

                                #2c. chercher un adjectif coordonné pour un attribut adjectival
                                if headchild.pos_ == 'ADJ' and headchild.dep_ == "conj": 
                                    conjheaddep = headchild.children                                                                                  
                                    for cchild in conjheaddep:
                                        if cchild is not None:
                                            neg_conj = False
                                            #2d. chercher un modifieur adverbial négatif pour un adjectif coordonné
                                            if cchild.dep_ == 'advmod':
                                                if cchild.text in negation.keys(): 
                                                    neg_conj = True                                                
                                                else: 
                                                    neg_conj = False  

                                    #écrire un adjectif coordonné et son modifieur adverbial négatif 
                                    if neg_conj == True: 
                                        triplet = [aspect, term, cchild.text + '_' + headchild.text] #NEG_CONJ
                                        triplets.append(triplet)
                                    else: 
                                        triplet = [aspect, term,  headchild.text] #CONJ
                                        triplets.append(triplet) 

                        #écrire un attribut adjectival et son modifieur adverbial négatif 
                        if neg_nsubj == True: 
                            triplet = [aspect, term,  headchild.text + '_' + token.head.text] #NEG_NSUBJ
                            triplets.append(triplet)
                        else: 
                            triplet = [aspect, term,  token.head.text] #NSUBJ
                            triplets.append(triplet)

            #sauvegarder seulement les triplets complets 
            if len(triplets) > 0: 
                print(triplets)
                resultat = {'phrase': sent.text, 'triplets': triplets}
                resultats.append(resultat)                   
                #print(resultats)

    #sauvegarder les triplets dans un fichier json
    with open('data.json', 'w', encoding='utf-8') as file:
        json.dump(resultats, file, indent=4)       
                


















