import pandas as pd
import numpy as np  # for data manipulation

from sklearn.model_selection import train_test_split  # for splitting the data into train and test
from sklearn.metrics import classification_report  # for model evaluation metrics
from sklearn import tree  # for decision tree model
import matplotlib.pyplot as plt
import math
import random

MIN = -1000
MAX = 1000
NUM_ARVORES = 40
NUM_AMOSTRAS_INTERVALO = 20


class Value:

        def __init__(self, value1, value2, numPerClass) -> None:
            self.value1 = value1
            self.value2 = value2
            self.count = 0
            self.numPerClass = numPerClass


class Column:

        def __init__(self, name, numAmostras, values) -> None:
            self.name = name
            self.numAmostras = numAmostras
            self.values = values


class No:

        def __init__(self, etiqueta, folha) -> None:
            self.etiqueta = etiqueta
            self.folha = folha
            self.filhos = []


def discretizar(df, columnsNames):
        
        

        #dfTest = df[1000:1499]
        #dfTraining = df[0:999]
        columns = []

        for col in columnsNames:

            #df[col].plot()
            #plt.show()

            if col == "classe":
                break

            values = []
            

            df = df.sort_values(col)

            gap = NUM_AMOSTRAS_INTERVALO
            sum = gap

            #menor = df[0:1]
            menorValor = MIN

            iteracoes = int(len(df)/gap)

            for i in range(0, iteracoes):
                intervalo = df[sum - gap:sum]

                maior = intervalo[gap - 1:gap]

                maiorValor = maior[col].values[0]

                #tam = maiorValor - menorValor
                #print(tam)
                #x = tam / 2

                #value1 = str(menorValor) + "/" + str(menorValor + x - 1)
                #value2 = str(menorValor + x) + "/" + str(maiorValor)

                #intervalo[col] = np.where(intervalo[col] < menorValor + x, value1, value2)
                #df.loc[sum-300:sum] = intervalo
                if i == (iteracoes-1):
                    maiorValor = MAX

                sum = sum + gap
                values.append(Value(menorValor, maiorValor, []))

                menorValor = maiorValor

            columns.append(Column(col, len(df), values))

        return columns


def escolherAtributo(df, columns):
        menorEntropia = 100000
        menorEntropiaColumn = columns[0]
        for col in columns:
            if col.name == "classe":
                break

            sumEntropiaColumn = 0
            #print(col.name + '\n')
            for i in range(0, len(col.values)):
                #print(col.values[i].value1)
                #print(col.values[i].value2)
                df2 = df[df[col.name] >= col.values[i].value1]
                df2 = df2[df2[col.name] < col.values[i].value2]
                #print(df2)

                col.values[i].count = len(df2)

                if len(df2) == 0:
                    continue

                value = col.values[i]
                sumEntropiaValor = 0
                
                for j in range(1, 5):
                    value.numPerClass.append(len(df2[df2["classe"] == j]))
                    prob = value.numPerClass[j - 1] / value.count
                    if prob != 0:
                        sumEntropiaValor = sumEntropiaValor - (
                            (prob) * math.log2(prob))
                sumEntropiaColumn =+ (value.count /
                                      col.numAmostras) * sumEntropiaValor

            if (menorEntropia > sumEntropiaColumn):
                menorEntropia = sumEntropiaColumn
                menorEntropiaColumn = col

        return menorEntropiaColumn

def id3(df, columns):
        biggerLength = 0
        biggerLenghtClass = 0
        for i in range(1, 5):
            length = len(df[df["classe"] == i])
            if length == len(df):
                return No([i], True)
            if length > biggerLength:
                biggerLength = length
                biggerLenghtClass = i

        if len(columns) == 0:
            return No([biggerLenghtClass], True)
        else:
            column = escolherAtributo(df, columns)

            for i in range(0, len(columns)):
                if columns[i].name == column.name:
                    columns.pop(i)
                    break

            raiz = No(column.name, False)
            for value in column.values:
                df2 = df[df[column.name] >= value.value1]
                df2 = df2[df2[column.name] < value.value2]
                raiz.filhos.append([value.value1, value.value2, id3(df2, columns)])

            return raiz


def testar(raizes, df):
        length = len(df)
        acertos = 0
        erros = 0
        votacao = []
        for i in range(0, 5):
            votacao.append(0)

        naoClassificavel = 0
        for i in range(0, length):
            amostra = df[i:i + 1]
            
            for j in range(0, len(raizes)):

                no = raizes[j]
    
                verif = 0
                while not no.folha and verif != len(no.filhos):
                    verif = 0
                    for filho in no.filhos:
                        if amostra[no.etiqueta].values[0] >= filho[0] and amostra[
                                no.etiqueta].values[0] < filho[1]:
                            no = filho[2]
                            break
                        verif = verif + 1
                if not no.folha:
                    naoClassificavel = naoClassificavel + 1
                else:
                    votacao[no.etiqueta[0]] = votacao[no.etiqueta[0]] + 1

            maisVotado = -1
            maiorNumVotos = -1
            for j in range(1, len(votacao)):
                if votacao[j] > maiorNumVotos:
                    maisVotado = j
                    maiorNumVotos = votacao[j]
                votacao[j] = 0
            if amostra["classe"].values[0] == maisVotado:
                acertos = acertos + 1
            else:
                erros = erros + 1
        
        print("Acertos: ", acertos)
        print("Erros: ", erros)
        print("Acurácia: ", acertos/(acertos + erros))
        print("Não classificáveis: ", naoClassificavel)


def main():
        pd.options.display.max_rows = 50
        df = pd.read_csv('treino_sinais_vitais_com_label.txt', encoding='utf-8')

        df = df.drop("id", axis=1)
        df = df.drop("pSist", axis=1)
        df = df.drop("pDiast", axis=1)
        df = df.drop("gravidade", axis=1)

        #df.plot(y="classe", x="bpM", kind="bar")
        #plt.show()



        dfTest = df[1000:1500]
        dfTraining = df[0:1000]
        raizes = []
        columnsNames = df.columns
        columnsNames = np.where(columnsNames != "classe", columnsNames, "bpM")
        partialColumnsNames = ['a', 'b']

    
        for i in range(0, NUM_ARVORES):


            randomIndex = random.randint(0, len(columnsNames) - 2)

            
            partialColumnsNames[0] = str(columnsNames[randomIndex])
            secondColumn = columnsNames[random.randint(0, len(columnsNames) - 1)]

            while secondColumn == partialColumnsNames[0]:
                secondColumn = columnsNames[random.randint(0, len(columnsNames) - 1)]

            partialColumnsNames[1] = str(secondColumn)
            #print(partialColumnsNames)

            dfPartialTraining = dfTraining.sample(frac=(1/NUM_ARVORES))
    
            columns = discretizar(dfPartialTraining, partialColumnsNames)
    
            #dfTest = df[1000:1500]
            raizes.append(id3(dfPartialTraining, columns))

        #dfTest = df.sample(frac=(1/3))
        testar(raizes, dfTest)

        #print(df)

        #print(len(dfTraining))


main()
