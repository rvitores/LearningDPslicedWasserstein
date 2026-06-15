import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from matplotlib.patches import Wedge

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    A GARDER
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def Cpt_DI(S,Y_pred,w=1 ,alpha=0.05, boxplot=False, wedge=False):
    '''
    Calculate the disparate impact and possibly return the wanted visualisation
    input:
        S: sensible variable
        Y_pred: predicted class
        1-alpha: quartile for the desired confidence interval
        w: if given, np.array containing the weight given to each observation
        boxplot/wedge: Optionally represent a boxplot and/or a wedge of the DI or the EoO
    output:
        Exact value, confidence interval of the
    '''

    n=S.shape[0]

    assert 0<alpha<2, 'alpha is the argument signifying the quartile, please let it be strictly between 0 and 2'
    assert n==Y_pred.shape[0],'every data must have the same length'

    if type(w) == int:
        Z = np.array([np.multiply(1-S,Y_pred),np.multiply(S,Y_pred),1-S,S])
    else:
        Z = np.array([np.multiply(w,np.multiply(1-S,Y_pred)),np.multiply(w,np.multiply(S,Y_pred)),np.multiply(w,1-S),np.multiply(w,S)])

    return _Asymptotic_behavior(Z,n,alpha, 'DI', boxplot, wedge)


def Cpt_EoO(S,Y_pred,Y_true,w=1 ,alpha=0.05, boxplot=False, wedge=False):
    '''
    Calculate the Equality of Odds index and possibly return the wanted visualisation
    input:
        S: sensible variable
        Y_pred: predicted class
        Y_true: true value to predict
        w: if given, np.array containing the weight given to each observation
        1-alpha: quartile for the desired confidence interval
        boxplot/wedge: Optionally represent a boxplot and/or a wedge of the DI or the EoO
    output:
        Exact value and confidence interval of the EoO
    '''

    n=S.shape[0]

    assert 0<alpha<2, 'alpha is the argument signifying the quartile, please let it be strictly between 0 and 2'
    assert n==Y_pred.shape[0],'every data must have the same length'

    if type(w) == int:
        Z = np.array([np.multiply(1-S,Y_pred),np.multiply(S,Y_pred),1-S,S])
    else:
        Z = np.array([np.multiply(w,np.multiply(1-S,Y_pred)),np.multiply(w,np.multiply(S,Y_pred)),np.multiply(w,1-S),np.multiply(w,S)])

    Z_EoO=np.zeros((4,n))
    for i in range (4):
        Z_EoO[i]=np.multiply(Z[i],Y_true)
    return _Asymptotic_behavior(Z_EoO,n,alpha, 'EoO', boxplot, wedge)


def Cpt_Suf(S,Y_pred,Y_true,w=1 ,alpha=0.05, boxplot=False, wedge=False):
    '''
    Calculate the Equality of Odds index and possibly return the wanted visualisation
    input:
        S: sensible variable
        Y_pred: predicted class
        Y_true: true value to predict
        w: if given, np.array containing the weight given to each observation
        1-alpha: quartile for the desired confidence interval
        boxplot/wedge: Optionally represent a boxplot and/or a wedge of the DI or the EoO
    output:
        Exact value and confidence interval of the EoO
    '''

    n=S.shape[0]

    assert 0<alpha<2, 'alpha is the argument signifying the quartile, please let it be strictly between 0 and 2'
    assert n==Y_pred.shape[0],'every data must have the same length'

    if type(w) == int:
        Z = np.array([np.multiply(1-S,Y_true),np.multiply(S,Y_true),1-S,S])
    else:
        Z = np.array([np.multiply(w,np.multiply(1-S,Y_true)),np.multiply(w,np.multiply(S,Y_true)),np.multiply(w,1-S),np.multiply(w,S)])

    Z_EoO=np.zeros((4,n))
    for i in range (4):
        Z_EoO[i]=np.multiply(Z[i],Y_pred)
    return _Asymptotic_behavior(Z_EoO,n,alpha, 'EoO', boxplot, wedge)


def _Asymptotic_behavior(Z,n,alpha, typ, boxplot, wedge):
    #Covariance matrix and expected value
    E, cov_matrix = _create_cov_matrix_and_esp(Z)

    #phi gradient applied to E
    grad_phi_E_T = [E[3]/(E[1]*E[2]) , -(E[0]*E[3])/(E[1]**2*E[2]) , -(E[0]*E[3])/(E[1]*E[2]**2) , E[0]/(E[1]*E[2])]

    #confidence interval
    Center, IC = _create_IC(n,E,cov_matrix, grad_phi_E_T, alpha, typ, boxplot, wedge)
    return Center, IC

def _create_IC(n,E,cov_matrix, grad_phi_E_T, alpha, typ, boxplot, wedge):
    Center = (E[0]*E[3])/(E[1]*E[2])
    sigma = np.dot(grad_phi_E_T, np.dot(cov_matrix, np.transpose(grad_phi_E_T)))
    norm_quartile = scipy.stats.norm.ppf(1-alpha/2)
    inter = (sigma/n)**(1/2)
    radius = inter*norm_quartile
    IC = [Center-radius, Center+radius]
    if boxplot:
        norm_quartile_25 = scipy.stats.norm.ppf(1-0.5/2)
        radius_25 = inter*norm_quartile_25
        data = [ IC[0],Center-radius_25, Center, Center+radius_25, IC[1] ]
        fig1, ax1 = plt.subplots()
        title = 'Boxplot at ' + str((1 - alpha)*100) +'% confidence interval of ' + typ
        ax1.set_title(title)
        ax1.boxplot(data)
    if wedge:
        _plot_wedge(IC, alpha, typ)
    return Center, IC



def _create_cov_matrix_and_esp(Z):
    E = np.zeros(4)
    cov_diag = np.zeros(4)
    cov_trig = np.zeros((4,4))
    for i in range (4):
        E[i] = np.mean(Z[i])
        cov_diag[i] = np.mean(np.multiply(Z[i],Z[i])) - E[i]**2
        for j in range(i): #must create the triangular sup, if not E won't be learnt yet
            cov_trig[i,j] = np.mean(np.multiply(Z[i],Z[j])) - np.multiply(E[i],E[j])
    return E, np.diag(cov_diag) + cov_trig + np.transpose(cov_trig)



def _plot_wedge(IC, alpha, typ):
    if IC[1]<1:
        plt.rcParams["figure.figsize"] = [5, 3]
    else:
        plt.rcParams["figure.figsize"] = [5, 5]
    fig, ax = plt.subplots()
    theta1, theta2 = IC[0]*180, IC[1]*180
    radius = 1
    center = (0, 0)
    w1 = Wedge(center, radius, 180, 180 - theta2, fc='white', edgecolor='black')
    w2 = Wedge(center, radius, 180 - theta2, 180 - theta1, fc='green', edgecolor='black', alpha=0.3, label="confidence interval of "+typ)
    if IC[1]<1:
        w3 = Wedge(center, radius, 180 - theta1, 180, fc='white', edgecolor='black')
    else:
        w3 = Wedge(center, radius, 180 - theta1, 180, fc='white', edgecolor='black')
    w4 = Wedge(center, radius, 0.2*180-0.3, 0.2*180+0.3, fc='darkred', label = '4/5 recommanded')
    t1 = plt.text(-1, 0, '0')
    t2 = plt.text(-2**(1/2)/2, 2**(1/2)/2, '0.25')
    t3 = plt.text(0, 1, '0.5')
    t4 = plt.text(2**(1/2)/2, 2**(1/2)/2, '0.75')
    t5 = plt.text(1, 0, '1')
    points = plt.scatter([-1, -2**(1/2)/2, 0, 2**(1/2)/2, 1], [0, 2**(1/2)/2, 1, 2**(1/2)/2, 0], color='black')

    for wedge in [w1, w2, w3, w4, t1, t2, t3, t4, t5, points]:
        ax.add_artist(wedge)

    ax.axis('equal')
    ax.set_xlim(-1.2, 1.2)
    if IC[1]<1:
        ax.set_ylim(-0.2, 1.2)
    else:
        t6 = plt.text(2**(1/2)/2, -2**(1/2)/2, '1.25')
        t7 = plt.text(0, -1, '1.5')
        t8 = plt.text(-2**(1/2)/2, -2**(1/2)/2, '1.75')
        points_2 = plt.scatter([2**(1/2)/2, 0, -2**(1/2)/2], [-2**(1/2)/2, -1, -2**(1/2)/2], color='black')
        for p in [t6, t7, t8, points_2]:
            ax.add_artist(p)
        ax.set_ylim(-1.2, 1.2)
    title = 'Wedge of ' + str((1 - alpha)*100) +'% Confidence about ' + typ +' criteria'
    ax.set_title(title)
    ax.legend()

    return fig


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    A VIRER ???
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
from matplotlib import colors

from scipy.stats import norm
from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier



def decorate_fairness_metric(function) :
    '''
    Cette fonction est la fonction décorateur des fonctions d'indices de fairness.
    La fonction qu'elle contient représente le traitement que l'on retrouve dans chacun des deux indices
    '''
    def fairness_metric(sensitive, prediction, observation=1, alpha=0.05) :
        '''
        Les fonctions d'indices de fairness ont pour entrée trois tableaux de même taille.
        Le premier contient les valeurs des variables sensibles.
        Le deuxième contient les valeurs des prédictions de l'algorithme de machine learning controlé.
        Le troisième contient les valeurs des variables réponses.
        Ce troisième tableau n'est pas nécessaire pour le disparate impact.

        Le paramètre alpha permet de modifier le niveau de confiance de l'intervalle de confiance.
        Il est paramétré par défaut à un niveau de confiance à 95%

        Les fonctions d'indices de fairness retournent un tableau de trois valeurs qui sont respectivement :
        La borne inférieure de l'intervalle de confiance de l'estimateur de l'indice de fairness.
        La valeur estimée de l'indice de fairness.
        La borne supérieure de l'intervalle de confiance de l'estimateur de l'indice de fairness.
        '''
        def h(x):
            return x[0] * x[3] / (x[1] * x[2])

        def grad_h(x):
            return np.array([x[3] / (x[1] * x[2]),
                             -x[0] * x[3] / ((x[1] ** 2) * x[2]),
                             -x[0] * x[3] / ((x[2] ** 2) * x[1]),
                             x[0] / (x[2] * x[1])])

        n = sensitive.shape[0]
        phi = function(sensitive, prediction, observation)
        fairness_metric = h(phi)
        grad = grad_h(phi)
        cov_4 = np.zeros((4, 4))
        cov_4 += np.array([[0, -phi[0] * phi[1], phi[3] * phi[0], -phi[3] * phi[0]],
                           [0, 0, -phi[2] * phi[1], phi[2] * phi[1]],
                           [0, 0, 0, -phi[2] * phi[3]],
                           [0, 0, 0, 0]])
        cov_4 += cov_4.T
        cov_4 += np.array([[phi[0] * (1 - phi[0]), 0, 0, 0],
                           [0, phi[1] * (1 - phi[1]), 0, 0],
                           [0, 0, phi[2] * phi[3], 0],
                           [0, 0, 0, phi[2] * phi[3]]])
        sigma = np.sqrt(np.dot(grad, np.dot(cov_4, grad.T)))
        z = norm.ppf(1 - alpha / 2, loc=0, scale=1)
        radius = (sigma * z) / np.sqrt(n)
        return np.array([fairness_metric - radius, fairness_metric, fairness_metric + radius])
    return fairness_metric


@decorate_fairness_metric
def disparate_impact(sensitive, prediction, observation=1, alpha=0.05) :
    '''
    Cette fonction utilise la fonction décorateur vue ci-dessus, et elle y ajoute le traitement spécifique au disparate impact.
    Pour obtenir un retour sur l'indice disparate impact, c'est cette fonction qu'il faut appeler.
    Le paramétrage est le même que la fonction décorateur.
    '''
    n = sensitive.shape[0]
    pi_1 = np.sum(sensitive) / n
    pi_0 = 1 - pi_1
    p_1 = np.sum(sensitive * prediction) / n
    p_0 = np.sum((1 - sensitive) * prediction) / n
    return [p_0, p_1, pi_0, pi_1]

@decorate_fairness_metric
def equality_of_odds(sensitive, prediction, observation, alpha=0.05) :
    '''
    Cette fonction utilise la fonction décorateur vue ci-dessus, et elle y ajoute le traitement spécifique a l'equality of odds.
    Pour obtenir un retour sur l'indice equality of odds, c'est cette fonction qu'il faut appeler.
    Le paramétrage est le même que la fonction décorateur.
    '''
    n = sensitive.shape[0]
    r_1 = np.sum(sensitive * observation) / n
    r_0 = np.sum((1 - sensitive) * observation) / n
    p_1 = np.sum(sensitive * prediction * observation) / n
    p_0 = np.sum((1 - sensitive) * prediction * observation) / n
    return[p_0, p_1, r_0, r_1]

def presentation_dataset(dataset) :
    '''
    Cette fonction prend en entrée le jeu de données à présenter.
    Elle ne produit aucun retour.
    Elle affiche un graphique présentant les individus du jeu de données en fonctions de leurs valeurs pour x1 et x2.
    '''
    COULEURS = ['#ecc19c', '#1e847f']

    (couleur_pos,  couleur_neg) = ([COULEURS[i] for i in dataset['s'][dataset.y == 1]], [COULEURS[i] for i in dataset['s'][dataset.y == 0]])

    (x1_pos, x1_neg) = (dataset['x1'][dataset.y == 1], dataset['x1'][dataset.y == 0])
    (x2_pos, x2_neg) = (dataset['x2'][dataset.y == 1], dataset['x2'][dataset.y == 0])

    plt.scatter(x1_pos, x2_pos, c = couleur_pos, marker = "^", s = 20, label = "y=1")
    plt.scatter(x1_neg, x2_neg, c = couleur_neg, marker = "x", s = 15, label = "y=0")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Représentation des individus')
    plt.tight_layout()
    return None

def repartition_reponses(dataset) :
    '''
    Cette fonction prend en entrée le jeu de données à présenter.
    Elle ne produit aucun retour.
    Elle affiche un graphique présentant la répartition des réponses positives et négatives dans chacune des classes sensibles.
    '''
    reponse_positive = dataset[dataset.y == 1]['s']
    reponse_negative = dataset[dataset.y == 0]['s']

    plt.hist(x = [reponse_positive, reponse_negative],
            color = ['green', 'red'],
            label = ['Réponse positive', 'Réponse négative'])
    plt.title('Répartition des réponses \n pour chaque classe sensible')
    plt.xticks([0,1], ['Classe discriminée', 'Classe favorisée'])
    plt.legend()
    plt.tight_layout()
    return None

def estimation_di_data(dataset) :
    '''
    Cette fonction prend en entrée le jeu de données à évaluer.
    Elle ne produit aucun retour.
    Elle affiche l'estimation et l'intervalle de confiance pour le disparate impact associé au jeu de données.
    '''
    di = disparate_impact(dataset['s'], dataset['y'])

    print('Estimateur du disparate Impact : ' + str(di[1]) + '\n' +
          'Intervale de confiance à 95% : [' + str(di[0]) + ' , ' + str(di[2]) + ']')
    return None

def estimation_di_modele(dataset, modele) :
    '''
    Cette fonction prend en entrée le jeu de données et le modèle à évaluer.
    Elle ne produit aucun retour.
    Elle affiche l'estimation et l'intervalle de confiance pour le disparate impact associé aux prédictions.
    '''
    di = disparate_impact(dataset['s'], dataset[modele + '_pred'])

    print('Estimateur du disparate Impact : ' + str(di[1]) + '\n' +
          'Intervale de confiance à 95% : [' + str(di[0]) + ' , ' + str(di[2]) + ']')
    return None

def estimation_eoo_modele(dataset, modele) :
    '''
    Cette fonction prend en entrée le jeu de données et le modèle à évaluer.
    Elle ne produit aucun retour.
    Elle affiche l'estimation et l'intervalle de confiance pour l'equality of odds associé aux prédictions.
    '''
    eoo = equality_of_odds(dataset['s'], dataset[modele + '_pred'], dataset['y'])

    print("Estimateur de l'equality of odds : " + str(eoo[1]) + '\n' +
          'Intervale de confiance à 95% : [' + str(eoo[0]) + ' , ' + str(eoo[2]) + ']')
    return None

def knn_train_and_predict(trainset, dataset, nb_voisins=5) :
    '''
    Cette fonction prend en entrée le jeu de données d'entrainement du modèle, ainsi que le jeu de données sur lequel on veut produire des prédictions.
    Elle entraine le modèle KNN avant d'effectuer les prédictions pour le jeu de données voulu.
    Elle affiche l'accuracy pour les prédictions effectuées.
    Elle retourne les prédictions du modèle.
    '''
    knn_model = KNeighborsClassifier(n_neighbors = nb_voisins)
    knn_model.fit(trainset[['x1', 'x2']], trainset['y'])
    pred = knn_model.predict(dataset[['x1', 'x2']])
    print('Accuracy : ' + str(accuracy_score(dataset['y'], pred)))
    return pred


def gb_train_and_predict(trainset, dataset) :
    '''
    Cette fonction prend en entrée le jeu de données d'entrainement du modèle, ainsi que le jeu de données sur lequel on veut produire des prédictions.
    Elle entraine le modèle gradient boosting avant d'effectuer les prédictions pour le jeu de données voulu.
    Elle affiche l'accuracy pour les prédictions effectuées.
    Elle retourne les prédictions du modèle.

    #inspired by lgb_train_and_predict(trainset, dataset)
    '''
    lgb_model = GradientBoostingClassifier()
    lgb_model.fit(trainset[['x1', 'x2']], trainset['y'])
    pred = lgb_model.predict(dataset[['x1', 'x2']])
    print('Accuracy : ' + str(accuracy_score(dataset['y'], pred)))
    return pred


def disparate_impact_sample(sensitive,
                            observation,
                            sample_size : int,
                            nb_sample : int=10) :
    '''
    Cette fonction prend en entrée un tableau des variables sensibles, un tableau des variables explicatives,  une taille d'échantillon, et un nombre d'échantillons à produire.
    Elle retourne la liste des disparate impact d'un nombre d'échantillons choisi, et d'une taille choisie.
    '''
    x = np.array(range(len(sensitive)))
    np.random.shuffle(x)
    di_list = []
    for i in range(nb_sample) :
        np.random.shuffle(x)
        sensitive = sensitive[x]
        observation = observation[x]
        sample_sensitive = sensitive[:sample_size]
        sample_observation = observation[:sample_size]
        sample_di = disparate_impact(sample_sensitive, sample_observation)[1]
        di_list.append(sample_di)
    return di_list

def disparate_impact_boxplot(boxes : list,
                             labels : list,
                             train_disparate_impact : int) :
    '''
    Cette fonction prend en entrée des listes de disparate impact, des labels associés à chaque liste (pour indiquer la taille des échantillons), et une valeur du disparate impact estimé sur la totalité du trainset.
    Elle ne produit aucun retour.
    Elle affiche les boxplot des différentes estimations du disparate impact sur des échantillons de même taille, et ce pour différentes tailles choisies.
    '''
    plt.boxplot(boxes)
    plt.xticks(range(1, len(boxes) + 1), labels)
    plt.axhline(y = train_disparate_impact, color = 'red', linestyle = '--', label = "Disparate Impact on train data")
    plt.axhline(y = 0.8, color = 'green', linestyle = '-', label = "0.8 line")
    plt.ylim(0,2)
    plt.ylabel("Disparate Impact")
    plt.xlabel("Sample Size")
    plt.legend()
    plt.show()
    return None

def echantillon_di_variabilite(dataset, model) :
    di_100 = disparate_impact_sample(dataset['s'], dataset[model + '_pred'], sample_size = 100, nb_sample = 50)
    di_250 = disparate_impact_sample(dataset['s'], dataset[model + '_pred'], sample_size = 250, nb_sample = 50)
    di_500 = disparate_impact_sample(dataset['s'], dataset[model + '_pred'], sample_size = 500, nb_sample = 50)
    di_750 = disparate_impact_sample(dataset['s'], dataset[model + '_pred'], sample_size = 750, nb_sample = 50)
    di_1000 = disparate_impact_sample(dataset['s'], dataset[model + '_pred'], sample_size = 1000, nb_sample = 50)

    disparate_impact_boxplot([di_100, di_250, di_500, di_750, di_1000],
                             ['100', '250', '500', '750', '1000'],
                             disparate_impact(dataset['s'], dataset['y'])[1])
    return None


def perceptron_train(x, y, learning_rate = 0.1, n_iter = 100):
    '''
    Cette fonction prend en entrée les variables explicatives, les variables réponses, un learning rate, et un nombre d'itérations pour l'apprentissage
    Elle renvoie les poids et le biais associés à l'entrainement du perceptron.
    Elle entraine un modèle perceptron à l'aide d'une descente de gradient.
    '''
    def model(x, weight, bias):
        z = x.dot(weight) + bias
        activation = 1 / (1 + np.exp(-z))
        return activation
    def initialisation(x):
        weight = np.random.randn(x.shape[1], 1)
        bias = np.random.randn(1)
        return (weight, bias)
    def gradients(activation, x, y):
        d_weight = 1 / len(y) * np.dot(x.T, activation - y)
        d_bias = 1 / len(y) * np.sum(activation - y)
        return (d_weight, d_bias)
    def update(d_weight, d_bias, weight, bias, learning_rate):
        weight = weight - learning_rate * d_weight
        bias = bias - learning_rate * d_bias
        return (weight, bias)
    weight, bias = initialisation(x)
    for i in range(n_iter):
        activation = model(x, weight, bias)
        d_weight, d_bias = gradients(activation, x, y)
        weight, bias = update(d_weight, d_bias, weight, bias, learning_rate)
    return (weight, bias)

def perceptron_predict(x, weight, bias, proba=False):
    '''
    Cette fonction prend en entrée des variables explicatives, des poids, un biais, et un booléen qui indique si l'on souhaite obtenir les proba obtenues, ou les réponses obtenues.
    Elle renvoie les prédictions du modèle perceptron entrainé avec la fonction précédente (à l'aide des poids et du biais).
    '''
    def model(x, weight, bias):
        z = x.dot(weight) + bias
        activation = 1 / (1 + np.exp(-z))
        return activation

    activation = model(x, weight, bias)
    if proba :
        return activation
    else :
        return (activation >= 0.5).astype('int')

def perceptron(trainset, dataset) :
    '''
    Cette fonction prend en entrée un jeu d'entrainement, et un jeu de données à prédire.
    Elle entraine un modèle perceptron avec la première entrée et produit les prédictions pour la seconde entrée.
    Elle indique le score du modèle.
    Elle renvoie les poids, le biais, et les prédictions du modèle.
    '''

    weight, bias = perceptron_train(np.array(trainset[['x1', 'x2']]),
                                    np.array(trainset['y']).reshape(-1, 1),
                                    learning_rate = 0.1,
                                    n_iter = 5000)
    pred = perceptron_predict(dataset[['x1', 'x2']],
                              weight,
                              bias)
    print('Accuracy : ' + str(accuracy_score(dataset['y'], pred)))
    return weight, bias, pred

def frontiere_perceptron(dataset, weight, bias) :
    '''
    Cette fonction prend en entrée le jeu de données, le poids et le biais du modèle perceptron.
    Elle ne renvoie rien.
    Elle affiche la frontière de décision du modèle sur le graphique qui représente le jeu de données.
    '''
    x1 = np.linspace(0,8,100)
    x2 = (- weight[0][0] * x1 - bias[0]) / weight[1][0]
    plt.plot(x1, x2, c = 'g', label = 'Frontière de décision')
    presentation_dataset(dataset)
    return None

def pire_cas(dataset, weight, bias, gamma) :
    '''
    Cette fonction calcule la distribution des pires cas et la présente dans un nuage de points en fonction de x1 et x2.
    '''

    COULEURS = ['#ecc19c', '#1e847f']

    x = dataset[['x1', 'x2']]
    y = dataset['y']
    s = dataset['s']

    n = len(s)
    n1 = sum(s)
    n0 = n - n1
    t = []
    for i in range(n) :
        t_i = {}
        # On défini les individus de la nouvelle distribution à l'aide du résultat de la maximisation
        t_i['x1'] = (weight[0] * (s[i] / n1 - (1 - s[i]) / n0)) / (2 * gamma) + x['x1'][i]
        t_i['x2'] = (weight[1] * (s[i] / n1 - (1 - s[i]) / n0)) / (2 * gamma) + x['x2'][i]
        t_i['y'] = y[i]
        t_i['s'] = s[i]
        t.append(t_i)

    dataset_pire = pd.DataFrame(t)
    dataset_pire['x1'] = dataset_pire['x1'].astype('float')
    dataset_pire['x2'] = dataset_pire['x2'].astype('float')

    (x1_pos, x1_neg) = (dataset['x1'][dataset.y == 1], dataset['x1'][dataset.y == 0])
    (x2_pos, x2_neg) = (dataset['x2'][dataset.y == 1], dataset['x2'][dataset.y == 0])
    (t1_pos, t1_neg) = (dataset_pire['x1'][dataset_pire.y == 1], dataset_pire['x1'][dataset_pire.y == 0])
    (t2_pos, t2_neg) = (dataset_pire['x2'][dataset_pire.y == 1], dataset_pire['x2'][dataset_pire.y == 0])
    (couleur_t_pos,  couleur_t_neg) = ([COULEURS[i] for i in dataset_pire['s'][dataset_pire.y == 1]], [COULEURS[i] for i in dataset_pire['s'][dataset_pire.y == 0]])

    x1 = np.linspace(0,8,100)
    x2 = (- weight[0][0] * x1 - bias[0]) / weight[1][0]
    plt.plot(x1, x2, c = 'g', label = 'Frontière de décision')
    plt.scatter(x1_pos, x2_pos, c = couleur_t_pos, marker = ".", s = 15, alpha = 0.5)
    plt.scatter(x1_neg, x2_neg, c = couleur_t_neg, marker = ".", s = 10, alpha = 0.5)
    plt.scatter(t1_pos, t2_pos, c = couleur_t_pos, marker = "^", s = 20)
    plt.scatter(t1_neg, t2_neg, c = couleur_t_neg, marker = "x", s = 15)
    plt.show()

    return None

def repartition_pire_cas(dataset, weight, gamma) :
    '''
    Cette fonction affiche les histogrammes pour x1 et x2 dans la distribution originale et dans la distribution des pires cas
    '''
    x = dataset[['x1', 'x2']]
    y = dataset['y']
    s = dataset['s']

    n = len(s)
    n1 = sum(s)
    n0 = n - n1
    t = []
    for i in range(n) :
        t_i = {}
        # On défini les individus de la nouvelle distribution à l'aide du résultat de la maximisation
        t_i['x1'] = (weight[0] * (s[i] / n1 - (1 - s[i]) / n0)) / (2 * gamma) + x['x1'][i]
        t_i['x2'] = (weight[1] * (s[i] / n1 - (1 - s[i]) / n0)) / (2 * gamma) + x['x2'][i]
        t_i['y'] = y[i]
        t_i['s'] = s[i]
        t.append(t_i)

    dataset_pire = pd.DataFrame(t)
    dataset_pire['x1'] = dataset_pire['x1'].astype('float')
    dataset_pire['x2'] = dataset_pire['x2'].astype('float')

    fig = plt.figure(figsize = (10,4))

    ax1 = fig.add_subplot(1,2,1)
    ax1.hist([dataset['x1'], dataset_pire['x1']], color = ['#077b8a', '#5c3c92'], label = ['Distribution originale', 'Distribution des pires cas'], bins = 40)
    ax1.set_title('Répartition des valeurs prises\npar x1 selon la distribution')
    ax1.legend()

    ax2 = fig.add_subplot(1,2,2)
    ax2.hist([dataset['x2'], dataset_pire['x2']], color = ['#077b8a', '#5c3c92'], label = ['Distribution originale', 'Distribution des pires cas'], bins = 40)
    ax2.set_title('Répartition des valeurs prises\npar x2 selon la distribution')
    ax2.legend()

    plt.tight_layout()
    return None

def repartition_pire_cas_2d(dataset, weight, gamma) :
    '''
    Cette fonction affiche les histogrammes en 2d des répartitions originale et des pires cas.
    '''
    x = dataset[['x1', 'x2']]
    y = dataset['y']
    s = dataset['s']

    n = len(s)
    n1 = sum(s)
    n0 = n - n1
    t = []
    for i in range(n) :
        t_i = {}
        # On défini les individus de la nouvelle distribution à l'aide du résultat de la maximisation
        t_i['x1'] = (weight[0] * (s[i] / n1 - (1 - s[i]) / n0)) / (2 * gamma) + x['x1'][i]
        t_i['x2'] = (weight[1] * (s[i] / n1 - (1 - s[i]) / n0)) / (2 * gamma) + x['x2'][i]
        t_i['y'] = y[i]
        t_i['s'] = s[i]
        t.append(t_i)

    dataset_pire = pd.DataFrame(t)
    dataset_pire['x1'] = dataset_pire['x1'].astype('float')
    dataset_pire['x2'] = dataset_pire['x2'].astype('float')

    fig = plt.figure(figsize = (14,4))

    ax1 = fig.add_subplot(1,3,1)
    ax1.hist2d(dataset['x1'], dataset['x2'], bins = 40, cmap = 'Greens')
    ax1.set_title('Répartition des valeurs prises\npar x1 et x2 dans la distribution originale')

    ax2 = fig.add_subplot(1,3,3)
    ax2.hist2d(dataset_pire['x1'], dataset_pire['x2'], bins = 40, cmap = 'Purples')
    ax2.set_title('Répartition des valeurs prises\npar x1 et x2 dans la distribution des pires cas')

    ax3 = fig.add_subplot(1,3,2)
    ax3.hist2d(dataset['x1'], dataset['x2'], bins = 40, cmap ='Greens', alpha = 0.8)
    ax3.hist2d(dataset_pire['x1'], dataset_pire['x2'], bins = 40, cmap = 'Purples', alpha = 0.8)
    ax3.set_title('Répartition des valeurs prises\npar x1 et x2 dans les deux distributions')

    plt.tight_layout()
    return None
"""
