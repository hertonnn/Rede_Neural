import numpy as np
import pickle


# funções matemáticas 

def aplicaSigmoid(matriz):

    resultado = []
    for i in range(len(matriz)):
        linha = []
        for j in range(len(matriz[0])):
            x = matriz[i][j]
            sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x))) # aplicação segura da sigmoid
            linha.append(sig)
        resultado.append(linha)

    return np.array(resultado)

def dSigmoid(matriz):

    resultado = []
    for i in range(len(matriz)):
        linha = []
        for j in range(len(matriz[0])):
            elemento = matriz[i][j] * (1 - matriz[i][j])
            linha.append(elemento)
        resultado.append(linha)

        return np.array(resultado)

# iniciando a rede neural

class RedeNeural:
    def __init__(self, tamEntrada, tamOculta, tamSaida) -> None:
        
        self.tamEntrada = tamEntrada
        self.tamOculta = tamOculta
        self.tamSaida = tamSaida


        self.bias_eo = np.random.rand(self.tamOculta,1)
        self.bias_os = np.random.rand(self.tamSaida, 1)

        self.pesos_eo = np.random.rand(self.tamOculta, self.tamEntrada)
        self.pesos_os = np.random.rand(self.tamSaida, self.tamOculta)

        self.learning_rate = 0.1

    def treino(self, entrada, erro):


        ###FeedFoward###

        self.camada_oculta = self.pesos_eo.dot(entrada)
        self.camada_oculta = self.camada_oculta + self.bias_eo
        self.camada_oculta = aplicaSigmoid(self.camada_oculta)

        self.saida = self.pesos_os.dot(self.camada_oculta) 
        self.saida = self.saida + self.bias_os
        self.saida = aplicaSigmoid(self.saida)


        ### Backpropagation ###
        
        self.erro = erro
        d_saida = dSigmoid(self.saida)
        t_oculta = self.camada_oculta.transpose()

        correcao_oculta_s = np.multiply(d_saida, self.erro) * self.learning_rate

        # ajuste bias
        self.bias_os = (self.bias_os + correcao_oculta_s) 

        correcao_oculta_s = correcao_oculta_s.dot(t_oculta) 
        self.pesos_os = (self.pesos_os + correcao_oculta_s)

        
        
        t_pesos_s = self.pesos_os.transpose()
        oculta_erro = t_pesos_s.dot(self.erro)
        d_oculta = dSigmoid(self.camada_oculta)
        t_entrada = entrada.transpose()

        gradient_oculta = np.multiply(oculta_erro, d_oculta) * self.learning_rate

        self.bias_eo = (self.bias_eo + gradient_oculta)

        correcao_oculta_i = gradient_oculta.dot(t_entrada)
        self.pesos_eo = (self.pesos_eo + correcao_oculta_i)

    def usaSalvo(self):

        arquivo = open("pesos.pkl","rb")

        self.bias_eo = pickle.load(arquivo)
        self.bias_os = pickle.load(arquivo)
        self.pesos_eo = pickle.load(arquivo)
        self.pesos_os = pickle.load(arquivo)

        arquivo.close()

    def predicao(self, entrada):



        ###FeedFoward###

        self.camada_oculta = self.pesos_eo.dot(entrada)
        self.camada_oculta = self.camada_oculta + self.bias_eo
        self.camada_oculta = aplicaSigmoid(self.camada_oculta)

        self.saida = self.pesos_os.dot(self.camada_oculta) 
        self.saida = self.saida + self.bias_os
        self.saida = aplicaSigmoid(self.saida)

        return self.saida
    
    def salvaDados(self):

        
        with open("pesos.pkl","wb") as arquivo:

            pickle.dump(self.bias_eo, arquivo)
            pickle.dump(self.bias_os, arquivo)
            pickle.dump(self.pesos_eo, arquivo)
            pickle.dump(self.pesos_os, arquivo)
        
        arquivo.close()



        







