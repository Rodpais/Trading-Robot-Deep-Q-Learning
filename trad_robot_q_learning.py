# %% [markdown]
# # Robo Trade 2.0 Usando Deep Q-Learning
# 
# Segunda tentativa de criação de uma rede neural para prever oscilações no mercado de ação.
# 
# Desta vez tentaremos usar Deep Q-Learning, usando Q-Networks.
# 
# <hr>
# 
# ## Funcionamento
# 
# Nossa rede neural terá a seguinte configuração:
# 
# * **Entrada:** os últimos X estados escolhidos pela variável ```window_size```.
# * **Saídas:** Comprar, Vender, Esperar.
# 
# <hr>
# 
# ## Classes e suas funções
# ### **AI_Trader:**
# - **Construtor:**
# 
#   Inicializa o agente que atuará no nosso ambiente.
# 
# - **Model Builder:**
# 
#   Modela a arquitetura da nossa rede neural de acordo com nossas escolhas.
# 
# - **Trade:**
# 
#   Função de decide se o agente irá executar um previsão usando a Q-Network ou executará uma ação gananciosa aleatória.
# 
#   Esta também é a função que retorna a resposta final da rede neural (Comprar, Vender ou Esperar).
# 
# - **Batch_Trade:**
# 
#   Função que realiza o treinamendo do lote de memórias.
# 
# 
# 
# 

# %%
# EXECUTAR NA PRIMEIRA EXECUÇÃO!

# %%
#pip install yfinance

# %%
# EXECUTAR NA PRIMEIRA EXECUÇÃO!

# Bitcoin Hora a hora - Data download !gdown --id '1VQry5JMRcuZ_BStIX8-FB8n4zFuDNUjk'


# %%
# Imports
import requests
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as data_reader

from tqdm import tqdm_notebook, tqdm
from collections import deque

from pandas_datareader import data as pdr
import yfinance as yfin
import datetime

import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import layers

url = 'https://drive.google.com/uc?id=1VQry5JMRcuZ_BStIX8-FB8n4zFuDNUjk'
response = requests.get(url)

with open('bitcoin_data.csv', 'wb') as f:
    f.write(response.content)

debug = 1

# %%
# Defining our Deep Q-Learning Trader

class AI_Trader():  

# -----------------------------------------------------------------------

  # CONSTRUTOR

  def __init__(self, state_size, action_space=3, model_name="AITrader"):
    
    self.state_size = state_size # Tamanho da entrada da rede neural 
    self.action_space = action_space # Espaço de ação será 3, Comprar, Vender, Sem Ação (Tamanho da saída da rede neural)
    self.memory = deque(maxlen=2000) # Memória com 2000 posições. A função Deque permite adicionar elementos ao final, enquanto remove elementos do início.
    self.inventory = [] # Terá as comprar que já fizemos
    self.model_name = model_name # Nome do modelo para o Keras
    
    self.gamma = 0.95 # Parâmetro que ajudará a maximizar a recompensa
    self.epsilon = 1.0 # Taxa de aleatoriedade para atitudes ganacioas do algorítimo.
    self.epsilon_final = 0.01 # Taxa final reduzida
    self.epsilon_decay = 0.995 # Velocidade de decaimento da taxa

    self.model = self.model_builder() # Inicializa um modelo e de rede neural e salva na classe

# -----------------------------------------------------------------------

  # DEFININDO A REDE NEURAL

  def model_builder(self):
        
    model = tf.keras.models.Sequential()      
    model.add(layers.Dense(units=32, activation='relu', input_dim=self.state_size))
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dense(units=self.action_space, activation='linear')) # De maneira geral, teremos 3 saída na rede geral (número de espaços de ação)


    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001)); # Compilamos o modelo

    return model # Retornamos o modelo pela função.

# -----------------------------------------------------------------------

  # FUNÇÃO DE TRADE
  # Usa o Epsilon e um número aleatório para definir se usará um dado aleatório ou a previsão da rede.

  def trade(self, state):
    if(debug):{print('TRADE FUNCTION:')}

    if random.random() <= self.epsilon:
      if(debug):{print('Entrou - Random')}
      return random.randrange(self.action_space)

    if(debug):{print('Vai Treinar Modelo')}
    actions = self.model.predict(state)
    if(debug):{print('Actions = ', actions)}
    if(debug):{print('Actions Argmax = ', np.argmax(actions[0]))}

    return np.argmax(actions[0])

# -----------------------------------------------------------------------

  # LOTE DE TREINAMENTO

  # Definindo o modelo para treinamento do lote

  def batch_train(self, batch_size): # Função que tem o tamanho do lote como argumento

    batch = [] # Iremos usar a memória como lote, por isso iniciamos com uma lista vazia

    # Iteramos sobre a memória, adicionando seus elementos ao lote batch
    for i in range(len(self.memory) - batch_size + 1, len(self.memory)): 
      batch.append(self.memory[i])

    # Agora temos um lote de dados e devemos iterar sobre cada estado, recompensa,
    # proximo_estado e conclusão do lote e treinar o modelo com isso.
    for state, action, reward, next_state, done in batch:
      reward = reward

      # Se não estivermos no último agente da memória, então calculamos a
      # recompensa descontando a recompensa total da recompensa atual.
      if not done:
        reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

      # Fazemos uma previsão e alocamos à varivel target
      target = self.model.predict(state)
      target[0][action] = reward

      # Treinamos o modelo com o estado, usando a previsão como resultado esperado.
      self.model.fit(state, target, epochs=1, verbose=0)

    # Por fim decrementamos o epsilon a fim de gradativamente diminuir tentativas ganaciosas. 
    if self.epsilon > self.epsilon_final:
      self.epsilon *= self.epsilon_decay

# -----------------------------------------------------------------------


# -----------------------------------------------------------------------


# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
    


# %%
# Stock Market Data Preprocessing

# Definiremos algumas funções uteis

# Sigmoid
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Função para formatar texto
def stock_price_format(n):
  if n < 0:
    return "- # {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))

# Busca dados no Yahoo Finance
# Formato data = "yyyy-mm-dd"
def dataset_loader(stock_name, initial_date, final_date):

  yfin.pdr_override()

  dataset = pdr.get_data_yahoo(stock_name, start=initial_date, end=final_date)
  
  start_date = str(dataset.index[0]).split()[0]
  end_date = str(dataset.index[1]).split()[0]
  
  close = dataset['Close']
  
  return close

# %% [markdown]
# # State Creator
#
# Primeiro vamos traduzir o problema para um ambiente de aprendizado por reforço.
# 
# * Cada ponto no gráfico é um ponto flutuante que representa o valor no momento do tempo.
#
# * Devemos prever o que acontecerá no próximo período de tempo, usando umas das 3 possibilidades de ação: compra, venda ou sem ação (esperar)
#
# Inicialmente vamos usar uma janela de 5 estados anteriores, para tentar prever o próximo.
#
# ```windows_size = 5```
#
# Ao invés vez de prever valores reais para nosso alvo, queremos prever uma de nossas 3 ações.
#
# Em seguida, mudamos nossos estados de entrada para diferenças nos preços das ações, que representarão as mudanças de preços ao longo do tempo.
#
# %%
# State Creator


def state_creator(data, timestep, window_size):
    # O index inicial (starting_id) será o timestep (passos/dias que já foram dados)
    # menos o tamanho da janela, que serão os dias olhados para trás.
  starting_id = timestep - window_size + 1
  if debug:
      print('Timestep = ', timestep)
  if debug:
      print('Window_size = ', window_size)
  if debug:
      print("Starting id = ", starting_id)
  if starting_id >= 0:
    windowed_data = data[starting_id: timestep + 1]

    if(debug):{print("Entrou no >=0. Starting id = ", starting_id)}
    if(debug):{print("windowed_data = ", windowed_data)}

  else:
    windowed_data =- starting_id * [data[0]] + list(data[0:timestep + 1])

    if(debug):{print("Entrou no Else. Starting id = ", starting_id)}
    if(debug):{print("w_d = ", windowed_data)}

state = [] # Criou uma array vazia para o estado

if(debug):{print('Vai entrar no FOR de normalização:')}

for i in range(window_size - 1):
  if(debug):{print('windowed_data[i + 1] = ', windowed_data[i+1])}
  if(debug):{print('windowed_data[i] = ', windowed_data[i])}

  state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))

  if(debug):{print('state = ',state)}

return np.array([state])

# %%


# %%
# Loading a Dataset

# CONFIGURAÇÕES DE IMPORTAÇÃO DE DADOS

# NOME DA AÇÃO
STOCK_NAME = "WEGE3.SA"

# DATA INCIAL
INITIAL_DATE = "2021-01-01"

# DATA FINAL
today = datetime.date.today()
FINAL_DATE = today.strftime("%Y-%m-%d") # Escolhe a data final como hoje

data = dataset_loader(STOCK_NAME, INITIAL_DATE, FINAL_DATE);

data

# %%
# Training the Q-Learning Trading Agent

window_size = 10
episodes = 2

batch_size = 32
data_samples = len(data) - 1

trader = AI_Trader(window_size)
trader.model.summary()

# if(debug):{print('Entrou - Random')}

trader.epsilon = 1


# %%
debug = 1

state = state_creator(data, 0, window_size + 1)

total_profit = 0;
trader.inventory = []

t=14



# %%
trader.epsilon = 0.1


if(debug):{print('------ MAKE TRADE:')}

action = trader.trade(state)

if(debug):{print('ACTION = ', action)}

if(debug):{print('------ FIND NEXT STATE:')}

next_state = state_creator(data, t+1, window_size + 1)
reward = 0

# action = 1



if(debug):{print('------ TAKING ACTION:')}
if(debug):{print('Tader Inventory = ', trader.inventory)}


if action == 1: #Buying
  if(debug):{print('ACTION = 1 (BUY)')}

  trader.inventory.append(data[t])
  print("AI Trader bought: ", stock_price_format(data[t]))

elif action == 2 and len(trader.inventory) > 0: #Selling
  if(debug):{print('ACTION = 0 (SELL)')}

  buy_price = trader.inventory.pop(0)
  if(debug):{print('Buy Price = ', buy_price)}
  reward = max(data[t] - buy_price, 0)
  if(debug):{print('dat[t] = ', data[t])}
  if(debug):{print('Reward = ', reward)}

  total_profit += data[t] - buy_price
  if(debug):{print('Total Pofit = ', total_profit)}
  print("AI Trader sold: ", stock_price_format(data[t]), " Profit: " + stock_price_format(data[t] - buy_price) )


if t == data_samples - 1:
    done = True
else:
    done = False

if(debug):{print('------ SAVING MEMORY:')}

trader.memory.append((state, action, reward, next_state, done))
if(debug):{print('Memory = ', trader.memory)}


state = next_state

t = t+1

# Se o tamanho da memória for maior que o tamanho do lote que definimos
# Então vamos treinar a rede, passando o tamanho do lote como argumento
if(debug):{print('Memory Len= ', len(trader.memory))}

if len(trader.memory) > batch_size:
  trader.batch_train(batch_size)

# %%
trader.memory

# %%
a = [1,2,100,4,5,6,7]

print(np.argmax(a))

# %%


# %%
# Defining a Training Loop

# Vamos iterar sobre todos episódios



for episode in range(1, episodes + 1):

  print("Episode: {}/{}".format(episode, episodes))

  state = state_creator(data, 0, window_size + 1)

  total_profit = 0
  trader.inventory = []

  #  O loop de treinamento que será executado durante uma época inteira
for t in tqdm(range(data_samples)):

    # O IA executa a função trade, que responderá com a ação que deve ser tomada
    action = trader.trade(state)

    # já foi dfinido o próximo estado
    # note que o definimos com t+1, pois estamos  considerando o próximo.
    # o valor da açao no index da tabela de dados.
    next_state = state_creator(data, t+1, window_size + 1)
    # sem recompensas até agora
    reward = 0

# Sem ação
    if action == 0:
      # Apenas um print e Recompensa = 0
      print(" - Sem ação | Total de papeis no portfolio = ", len(trader.inventory))

    # Compra
    if action == 1: #Comprando
      # Recompensa = 0

      # Adicionamos a ação comprada na array de portfolio
      trader.inventory.append(data[t])

      print(" - AI Trader Comprou: ", stock_price_format(data[t]))

    # Venda (Deve possuir ações no portfolio)
    elif action == 2 and len(trader.inventory) > 0:   #vendendo

      # Remove última ação do portfólio e a retorna
      buy_price = trader.inventory.pop(0)

      # Recompensa = lucro ou 0 se houve prejuízo.
      reward = max(data[t] - buy_price, 0)

      total_profit += data[t] - buy_price # Soma ao lucro/prejuízo total

      print(" - AI Trader Vendeu: ", stock_price_format(data[t]), " - Lucro: " + stock_price_format(data[t] - buy_price) )


    # Verifica se estamos no final de uma época
    if t == data_samples - 1:
      done = True
    else:
      done = False


    # Salvamos os dados na memória, na mesma ordem que na função BATCH_TRAIN
    trader.memory.append((state, action, reward, next_state, done))

    # Definimos que o estado atual é o próximo estado calculado anteriormente
    state = next_state

    if done:
      print("########################")
      print("LUCRO TOTAL: {}".format(total_profit))
      print("########################")


    # Se o tamanho da memória for maior que o tamanho do lote que definimos
    # Então vamos treinar a rede, passando o tamanho do lote como argumento
    if len(trader.memory) > batch_size:
      trader.batch_train(batch_size)

  # A Cada 10 episódios treinados, salvamos a rede
    if episode % 10 == 0:
      trader.model.save("ai_trader_{}.h5".format(episode))

