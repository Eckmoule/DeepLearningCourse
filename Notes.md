# Notes

Ce document contient mes notes prises pour relecture. Il est à l'état de brouillon. 

### Métrique de performance

La métrique de performance est une métrique à destination de l’utilisateur ou du Datascientist qui va fournir le nombre de prédiction correctes et incorrectes. Dans l’exemple 
où l’on cherche à déterminer si l’image en entrée est un trois ou un sept on pourra dire sur les 10.000 images entrées dans le jeux 8.500 ont été correctement labellisés soit 
85% de taux de réussite.  

```python
def accuracy(activations, targets):
  predictions = activations > 0
  corrects = (predictions -- targets)
  return corrects.float().mean()
```

Cette métrique ne peut pas être utilisé par le machine learning pour s'entraîner car le résultat est binaire (bon/pas bon). Nous allons chercher une méthode permettant de savoir si le changement nous a fait nous rapprocher d’un bon résultat même s’il est toujours mauvais. 

### Fonction de coût

La fonction de coût (loss function) va servir à déterminer le niveau d’erreur de notre modèle afin de l’ajuster progressivement (à la différence de la métrique de performance 
ci-dessus). L’objectif d’une bonne fonction de coût est qu’une amélioration d’un paramètre fasse diminuer le coût de manière régulière. Ainsi ce signal pourra être utilisé pour 
continuer l’amélioration. 

La notion qui sera souvent utilisée est la notion de distance. A contrario d’un résultat binaire bon/pas bon la distance permet de déterminer si l’on est plus ou moins proche 
du bon résultat. Cette fonction va prendre en entrée les résultats du modèle ainsi que les labels à trouver (dans l’exemple 0 = faux, 1 = vrai). 

__Cas avec une seule catégorie (est un chien ?)__: Elle va appliquer la méthode sigmoid pour ramener les résultats à des valeurs entre 0 et 1 puis rendre la distance moyenne. 

__Cas avec plusieurs catégories et une seul vrai (quel race de chien ?)__: Dans le cas ou on a plusieurs catégorie la méthode pour ramener les nombre entre 0 et 1 sera softmax. 
Cette méthode permet de ramener tout les nombre entre 0 et 1 et de faire en sorte que leur somme fasse 1.

```python
def mnist_loss(activations_01, targets)
  activations_01.sigmoid()
  distances = torch.where(targets==1,1-activations_01,activations_01)
  return distances.mean()
```

### Utilisation de la fonction de coût (anticiper la variation de l’erreur)

il va nous falloir déterminer quel paramètre modifier dans quel sens pour améliorer le coût. L'approche "basique" consisterait à modifier un seul paramètre puis à recalculer 
le coût afin de déterminer si la modification est bénéfique ou non. Même avec le problème exemple "simple" nous avons déjà 784 paramètres et cela serait particulièrement 
coûteux. 

Dans les faits nous allons utiliser les principes mathématiques de dérivation/gradient (a voir). Le gradient permet de déterminer “si je modifie ce paramètre d’une petite 
quantité comment varie l’erreur”. 

On va demander à PyTorch de prévoir les gradients sur nos paramètres. 
```python
params.requires_grad()
```
Après avoir fait notre prédiction et calculé notre coût d’erreur 
```python
predictions = model(inputs,params)
loss = mean_squared_error(predictions,targets)
```
On appel la fonction backward qui va remonter les calculs dans le sens inverse (back propagation) et renseigner les gradients qui seront alors disponibles dans la propriétée 
grad. 
```python
loss.backward()
params.grad
```

Afin d’optimiser le calcul on va calculer les gradients sur une partie du jeux d'entraînement seulement. Si l’on a 6.000 exemples d'entraînement en entrée on peut décider de 
réaliser un seul pas d’ajustement sur les 6.000 ou bien 60 pas en utilisant des blocs de 100 exemples. 
La taille de ce “bloc” est un compromis entre un nombre d’exemple suffisamment grand pour qu’il soit représentatif mais suffisamment petit pour optimiser le traitement. 

Pour cela nous allons utiliser la classe DataLoader de FastAI2
```python
coll = tensor([1,2,3,4,5,6,7,8,9,10])
dl = DataLoader(coll,batch_size=3,shuffle=True
```
Ce que nous allons vouloir mettre en mini-batch est la datasource qui correspond à un ensemble de couple valeur / labels.

### Le chargement de données - DataBlock

L’objectif est de disposer de deux grands tableaux de nombres (les éléments d’entrée et les labels attendus en sortie). 
Nous allons utiliser l’objet DataBlock pour cela, la mécanique est la suivante: 

L’objet prend une sources (répertoire ou fichier) et exécute: 

1. Une méthode get_items pour récupérer les chemins ou les lignes 
2. Parcourir un par un les items et applique get_x (ce que l’on attend en entrée - données) et get_y (ce que l’on attend en sortie - Label)
3. Définir des Blocks qui vont permettre de définir le type de données pour le x et le y (Images et Catégorie par exemple). 
4. Une méthode item_tfms qui va permettre de transformer chaque item (entrée et sortie possible). On peut par exemple redimensionner les images pour qu’elles aient toutes la 
même taille
5. Une méthode split pour partager les données entre jeux d'entraînement et de validation. 
6. Une méthode batch_tfms qui va me permettre de regrouper mes données par batch (en lui passant la taille d’un batch).

Il est possible d’utiliser un exemple de Block fournies par défault par fastAI (https://dev.fast.ai/data.transforms)

### Fonction de coût:

#### Cross entropy 

Cette fonction est généralement utilisée pour les problématiques de catégorisation. On va chercher à déterminer à quelle catégorie notre entrée (image, ligne de données, …) 
appartient. 

Le système de label utilisé pour ce type de problème est appelé one-hot encoding. On ne va pas donner un nombre à chaque catégorie (0 = Chien, 1 = Lézard, 2 = Chèvre, …) et 
tenter de le déterminer. Cette approche ne convient pas car elle indique au modèle mathématique un écart de 2 entre chien et chèvre et un écart de 1 entre chien et lézard, ce 
qui n’a aucun sens. 
Le principe va plutôt être de déterminer de façon binaire est-ce que c’est un chien (0 ou 1), est-ce que c’est un lézard (0 ou 1), …. On aura donc autant de nombre (0 ou 1) en 
sortie que de catégorie. 

Pour obtenir cet ensemble de nombre le modèle sera appliqué aux données d’entrée et on obtiendra autant de nombre (entre -infini et infini) que de catégorie. On va ensuite 
appliquer la méthode mathématique softmax pour ramener ces nombres entre 0 et 1 et faire en sorte que leur somme fasse 1 (contrairement à sigmoid). 

NB: Attention softmax utilise les exponentiels et va donc accentuer les écarts entre les prédictions initiales. C’est plutôt “positif” dans notre cas car cela aide à détacher 
une prédictions des autres mais peut nous induire en erreur. 

Dans le cas ou l’on a une seule catégorie à déterminer on va s’intéresser à la distance entre l’activation pour cette catégorie (entre 0 et 1) calculée à l’étape précédente 
et 1. 
Par exemple on pourrait avoir en sortie de notre modèle: 

Chien (à trouver) | Lézard |  Chèvre
----------------- | ------ |  ------
85.23 | -2.55 | -45.36

On applique softmax pour normaliser (ramener entre 0 et 1 et somme = 1)

Chien (à trouver) | Lézard |  Chèvre
----------------- | ------ |  ------
0.8 | 0.1 | 0.1

La distance serait donc de 0.2 (1 - 0.8). 

En se rapprochant de 1 la diminution de distance va devenir de plus en plus petite alors que l’amélioration réelle du modèle est la même. J’ai divisé l’erreur par deux en 
passant d’une distance de 0.5 à 0.25 (gain 0.25) ou d’une distance de 0.1 à 0.05 (gain de 0.05). 
Pour palier à ce problème et obtenir une amélioration constante nous allons utiliser la fonction mathématique log(). En comparant le log de la distance l’amélioration sera 
similaire pour une division par 2 de l’erreur par exemple. 

log(0.5) - log(0.25) = 0.3
log(0.1) - log(0.05) = 0.3 

NB: Le nom mathématique de ce calcul est “negative log likelihood”

Ainsi on a donc la même amélioration (0.3) pour les deux division par 2 de notre distance. Cette distance sera rendue par la fonction de coût et permettra l'entraînement.

L’utilisation de softmax pour normaliser les activations (ramener entre 0 et 1 et faire en sorte que leurs somme valent 1) et de negative log likelihood pour la distance 
correspond à la fonction de coût appelée __Cross Entropy__. 

### Transfer Learning

Il y a deux partie distincte des réseaux de neurones que l’on va utiliser:  

La première partie appelé corps (body ou backbone) est une partie pré-entraîné (sur imagenet dans notre exemple pour les images). Tout un ensemble de “neurones” ont donc été 
entraînés avec des machines très puissantes à extraire des caractéristiques utiles dans les images (forme, ). On va donc pouvoir utiliser cette partie telle quelle en “empruntant”les paramètres pré-entraînés (en effet le résultat de l'entraînement n’est “qu’un” ensemble de bon paramètres à passer au modèle pour qu’il détecte bien). 

NB: Cela s'appelle du transfer learning (on utilise les paramètres d’un autre entrainement

Une deuxième partie appelé tête (head) qui basée sur l’ensemble des informations obtenues précédemment va réaliser la classification (j’ai x caractèristiques et cela va me 
permettre de classifier si c’est une voiture, un humain, …).  Cette partie est spécifique à ce que l’on souhaite classifier contrairement à la première qui est plus “générique”. 

Durant l'entraînement on va d’abord entraîner la partie head qui est spécifique à notre problème. Dans un second temps on pourra essayer d’affiner (très légèrement) la partie 
body à notre problème afin qu’elle détecte plutôt ce que l’on cherche.

NB: On utilisera les termes de freezing/unfreezing pour indiquer si les parties sont entraînées ou pas. 

Techniquement on peut “débloquer” la partie body en utilisant la méthode Unfreeze 

```python
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.unfreeze()
```
 
On va utiliser une learning rate beaucoup plus importante pour entraîner la partie head (spécifique) que le body (générique). Il est également possible d’appliquer des learning 
rate différent pour les différents neurones de la première partie. En effet plus on est proche du début plus les détections sont génériques et ne nécessitent donc pas d’affinage
en fonction du problème. 

NB: Le fait d’appliquer des learning rate différent en fonction du “niveau” de spécialisation s’appel discriminative learning rate. 

Ce découpage est déterminé par une fonction qu’il est possible de passer à l’objet learner de FastAi si on ne souhaite pas utiliser la mécanique par défaut. 
