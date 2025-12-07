# **Avaliação entre Privacidade e Acurácia em Modelos de Deep Learning com DP-SGD**

Este repositório contém o código-fonte utilizado no artigo:

**Uma Avaliação entre Privacidade e Acurácia em Modelos de Aprendizado Supervisionado em Deep Learning com DP-SGD (_Differentially Private Stochastic Gradient Descent_)**

Autores:  
- **Gustavo Gobi Martinelli** – gustavomartinelli@gmail.com  
- **Rodolfo da Silva Villaça** – rodolfo.villaca@inf.ufes.br  

O estudo investiga empiricamente o *trade-off* entre **privacidade diferencial** e **acurácia** no treinamento de modelos de aprendizado supervisionado, utilizando o dataset CIFAR-10 e a técnica **DP-SGD**, implementada com a biblioteca **Opacus**.

O notebook completo está disponível neste repositório no arquivo **`avalia_priv_acc_ml-dp_dp-sgd.ipynb`**, bem como no Google Colab por meio do link abaixo (somente leitura):

## 👉 **Link do Colab => [avalia_priv_util_ml-dp_dp-sgd.ipynb](https://colab.research.google.com/drive/1NwFBL9jUxME8EfLso901TJIOXqBykD9P?usp=sharing)**

## **📌 Objetivo do Projeto**

O propósito central deste experimento é demonstrar, de forma reprodutível:

1. Como a injeção de ruído Gaussiano no processo de treinamento preserva privacidade sob o paradigma de **_Differential Privacy_**.  
2. Como diferentes níveis de ruído afetam a **acurácia**, a **convergência**, e o **ε (epsilon)** consumido.  
3. A comparação direta entre:
   - **Treinamento tradicional (sem privacidade)**  
   - **Treinamento com DP-SGD** nos cenários:
     - Privacidade Fraca  
     - Privacidade Média  
     - Privacidade Forte  

As análises incluem curvas de *loss*, acurácia, evolução do ε, matrizes de confusão e gráficos de *trade-off*.

## 📂 **Estrutura do Repositório**
```
avalia_priv_acc_ml-dp_dp-sgd/
│
├── README.md
├── avalia_priv_acc_ml-dp_dp-sgd.ipynb   # Notebook principal
```

## **🧪 Descrição do Experimento**

O código presente no notebook executa:

### **1) Carregamento do CIFAR-10**
Conversão para tensores e inicialização dos *dataloaders* de treino e teste.

### **2) Definição do Modelo (CNN personalizada)**
Modelo `DPCNN`, com camadas convolucionais, *GroupNorm*, *ReLU*, *MaxPool* e dois classificadores lineares.

### **3) Treinamento sem Privacidade (Baseline)**
Será adicionada uma célula que treina o modelo com SGD **sem** DP-SGD, para permitir comparação direta.

### **4) Treinamento com DP-SGD**
Utilização do **Opacus** com `noise_multiplier` variando conforme os cenários:

| Cenário               | Noise Multiplier (σ) |
|----------------------|----------------------|
| Privacidade Fraca    | 0.3                  |
| Privacidade Média    | 0.8                  |
| Privacidade Forte    | 1.5                  |

O código coleta:

- *Loss* por época  
- Acurácia por época  
- ε (epsilon) por época  
- Matriz de Confusão final  
- Resultados consolidados em `df_results`

### **5) Geração de Gráficos**
O notebook produz:

- Curva de *loss* por cenário  
- Curva de acurácia por cenário  
- Evolução do ε  
- Matriz de confusão para cada cenário  
- Gráfico ε × acurácia  
- Gráfico σ × acurácia  
- Gráfico σ × ε  
- Gráficos comparativos finais (barras)

## **📊 Resultados Esperados**

O treinamento DP-SGD demonstra empiricamente:

- **Quanto maior o ruído**, **maior a privacidade** (menor ε).  
- **Quanto maior a privacidade**, **menor a acurácia** — devido ao impacto do ruído no gradiente.  
- A qualidade dos modelos sem DP é superior, porém **não oferecem proteção formal contra ataques de inferência**.

Os resultados completos podem ser visualizados no notebook.

## **▶️ Como Executar o Notebook**

1. Abra o link do Google Colab.  
2. Selecione “Executar tudo”.  
3. Certifique-se de que a GPU está ativada no ambiente do Colab.  
4. Caso execute localmente:
   ```bash
   pip install opacus torch torchvision seaborn scikit-learn

## 🔒 **Sobre Privacidade Diferencial e DP-SGD**
O método DP-SGD, proposto inicialmente por **Abadi et al. (2016)**, aplica:
* **Clipping dos gradientes**
* **Ruído Gaussiano** proporcional ao nível de privacidade desejado
* **Rastreamento do ε** ao longo do treinamento
Este repositório demonstra a implementação prática e sua análise experimental.

## 📜 **Licença Recomendada**
A exigência é que **os autores sejam sempre mencionados.**
A licença que melhor atende esse requisito é:

## 👉 **Licença BSD 3-Clause**
Ela permite uso, modificação e redistribuição, desde que **o aviso de copyright seja mantido** — cumprindo exatamente sua exigência.

**_BSD 3-Clause License_**

_Copyright (c) 2025, Gustavo Martinelli & Rodolfo da Silva Villaça_

_Redistribution and use in source and binary forms, with or without_
_modification, are permitted provided that the following conditions are met:_

_1. Redistributions of source code must retain the above copyright notice, this_
   _list of conditions and the following disclaimer._

_2. Redistributions in binary form must reproduce the above copyright notice,_
   _this list of conditions and the following disclaimer in the documentation_
   _and/or other materials provided with the distribution._

_3. Neither the name of the copyright holder nor the names of its_
   _contributors may be used to endorse or promote products derived from_
   _this software without specific prior written permission._

_THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"_
_AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE_
_IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE_
_DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE_
_FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL_
_DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR_
_SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER_
_CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,_
_OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE_
_OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE._

## 📞 **Contato dos Autores**

**Gustavo Gobi Martinelli**

Email: [gustavomartinelli@gmail.com](gustavomartinelli@gmail.com)

**Prof. Rodolfo da Silva Villaça**

Email: [rodolfo.villaca@inf.ufes.br](rodolfo.villaca@inf.ufes.br)

## 📝 **Observação Final**

Este README serve como documentação pública do experimento e dos resultados apresentados no artigo.
Caso deseje contribuir, testar variações ou reportar _issues_, fique à vontade para abrir uma discussão no repositório.
