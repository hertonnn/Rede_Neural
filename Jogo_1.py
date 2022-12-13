import pygame
import random
import numpy as np 
from RedeNeural import *
import matplotlib.pyplot as plt # Criação de Gráficos 

PRETO=0,0,0
BRANCO=255,255,255
VERDE=0,255,0
VERMELHO=255,0,0

fim=False
tamanho=800,600
tela=pygame.display.set_mode(tamanho)
tela_retangulo=tela.get_rect()
tempo=pygame.time.Clock()
pygame.display.set_caption("Jogo Pong")

class Raquete:
    def __init__(self, tamanho, pos):
        self.imagem=pygame.Surface(tamanho)
        self.imagem.fill(BRANCO)
        self.imagem_retangulo=self.imagem.get_rect()
        self.velocidade = 30
        self.imagem_retangulo[0] = pos

    def move(self, y):

        self.imagem_retangulo[1] += y * self.velocidade

        # máquina

    def atualiza(self, tecla):
        if tecla[0][0] > 0.5:
            self.move(-1)
        if tecla[0][0] < 0.5: 
            self.move(1)
        self.imagem_retangulo.clamp_ip(tela_retangulo)

        # jogador 

    def atualiza_jogador(self, tecla):
        if tecla[pygame.K_UP]:
            self.move(-1)
        if tecla[pygame.K_DOWN]:
            self.move(1)
        self.imagem_retangulo.clamp_ip(tela_retangulo)

    def realiza(self):
        tela.blit(self.imagem, self.imagem_retangulo)

class Bola:
    def __init__(self, tamanho):
        self.altura, self.largura = tamanho
        self.imagem=pygame.Surface(tamanho)
        self.imagem.fill(BRANCO)
        self.imagem_retangulo=self.imagem.get_rect()
        self.velocidade = 10 # VELOCIDADE DA BOLA
        self.flag = False               
        self.set_bola()

    def aleatorio(self):
        while True:
            num=random.uniform(-1.0, 1.0)
            if num > -0.5 and num < 0.5:
                continue
            else:
                return num

    def set_bola(self):
        x=self.aleatorio()
        y=self.aleatorio()
        self.imagem_retangulo.x = tela_retangulo.centerx
        self.imagem_retangulo.y = tela_retangulo.centery
        self.velo=[x, y]
        self.pos = list(tela_retangulo.center)

    def colide_parede(self):
        if self.imagem_retangulo.y < 0 or self.imagem_retangulo.y > tela_retangulo.bottom - self.altura:
            self.velo[1] *= -1
            self.flag = False

               
        if self.imagem_retangulo.x < 0 or self.imagem_retangulo.x > tela_retangulo.right - self.largura:
            self.velo[0] *= -1
            if self.imagem_retangulo.x > tela_retangulo.right - self.largura:
                placar1.pontos_Jogador = 0
            if self.imagem_retangulo.x < 1 and self.flag == False:
                placar1.pontos = 0
                self.flag = True
                
                # global erro
                # erro = (raquete.imagem_retangulo.y - bola.imagem_retangulo.y)

    def colide_raquete(self, raquete_rect):
        if self.imagem_retangulo.colliderect(raquete_rect) and self.imagem_retangulo.x < 400:
            self.velo[0] *= -1
            placar1.pontos += 1  

            global erro
            erro = 0

        if self.imagem_retangulo.colliderect(raquete_rect) and self.imagem_retangulo.x > 400:
            self.velo[0] *= -1
            placar1.pontos_Jogador += 1        


    def move(self):
        self.pos[0] += self.velo[0] * self.velocidade
        self.pos[1] += self.velo[1] * self.velocidade
        self.imagem_retangulo.center = self.pos

    def atualiza(self, raquete_rect):
        self.colide_parede()
        self.colide_raquete(raquete_rect)
        self.move()

    def realiza(self):
        tela.blit(self.imagem, self.imagem_retangulo)


class Placar:
    def __init__(self):
        pygame.font.init()
        self.fonte =pygame.font.Font(None, 36)
        self.pontos = 0 
        self.pontos_Jogador = 0

    def contagem(self):
        self.text=self.fonte.render("IA  " + str(self.pontos) + "  |  " + str(self.pontos_Jogador) + "  Player", 1, (VERDE))
        self.textpos=self.text.get_rect()
        self.textpos.centerx=tela.get_width() / 2
        tela.blit(self.text, self.textpos)
        tela.blit(tela, (0, 0))


raquete = Raquete((10, 100), 0)
rede = RedeNeural(3, 3, 1)
raquete_jogador = Raquete((10, 100), 800)
bola=Bola((15, 15))
placar1=Placar()

# iniciar
MODO_JOGO = True
# contra a máquina(requer aumento de velocidade da polinha- linha 58)
MODO_CONTRA_IA =  False
# gráfioo em tempo real
MODO_GRAFICO =  False
# atualizar pesos da rede neural
MODO_TREINAMENTO = True 

x = []
y = []
frames = 0
limitador = 0 # limitar atualização de gráfico

if MODO_GRAFICO:
    plt.style.use('dark_background') # estilo
    plt.figure(figsize=(4, 2), label = 'Gráfico Acurácia IA') # Aparência do gráfico
    

if not MODO_TREINAMENTO:
    rede.usaSalvo()

# início
if MODO_JOGO == True:
    while not fim:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                fim=True
        
     
        # definindo dados para a rede neural
        entrada = [[raquete.imagem_retangulo.y / 600], [bola.imagem_retangulo.y / 600], [bola.imagem_retangulo.x / 800]]
        entrada = np.array(entrada)
        erro = (raquete.imagem_retangulo.y - bola.imagem_retangulo.y) / 10

        tecla = rede.predicao(entrada)
        tela.fill(PRETO)

        if MODO_CONTRA_IA:
            tecla_jogador = pygame.key.get_pressed()
            raquete_jogador.realiza()
            raquete_jogador.atualiza_jogador(tecla_jogador)
            bola.atualiza(raquete_jogador.imagem_retangulo)

        raquete.realiza()
        bola.realiza()
        raquete.atualiza(tecla)
        bola.atualiza(raquete.imagem_retangulo)
        tempo.tick(30)
        placar1.contagem()
        
        print(" Erro = " + str(erro))
        print(" decisão = " + str(tecla))

        if MODO_TREINAMENTO:
            rede.treino(entrada, erro)
        
        if MODO_GRAFICO and limitador == 30:
            plt.ion()

            plt.cla()
            plt.clf()
            x.append(erro)
            y.append(frames)
            plt.plot(y,x, label = 'Acurácia', color = 'green')
            plt.ylabel('Valor do erro')
            plt.xlabel('N• de Frames')
            plt.legend()
            plt.pause(0.1)

            limitador = 0
            plt.ioff()
        
        frames += 1
        limitador += 1
        pygame.display.update()

# salvar últimos pesos
if MODO_TREINAMENTO:
    rede.salvaDados()
