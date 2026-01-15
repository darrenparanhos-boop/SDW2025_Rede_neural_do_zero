# 🧠 Rede Neural do Zero com PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10Jb6rMh27h-tgDsE5Yx6iBM3oOMjsvlO?usp=sharing)

Este projeto apresenta a implementação de uma **rede neural totalmente conectada (MLP)** desenvolvida do zero utilizando **PyTorch** e o dataset **MNIST**.  
O estudo foi realizado no **Google Colab** e documentado no notebook [`Rede_neural_do_zero.ipynb`](https://colab.research.google.com/drive/10Jb6rMh27h-tgDsE5Yx6iBM3oOMjsvlO?usp=sharing).

---

## 📌 Objetivos
- Implementar uma rede neural simples para classificação de dígitos manuscritos.  
- Explorar conceitos fundamentais de **camadas lineares**, **funções de ativação** e **otimização**.  
- Demonstrar o fluxo completo: **pré-processamento → modelagem → treinamento → validação**.  

---

## 📚 Tecnologias Utilizadas
- **Python 3.14**
- **PyTorch**
- **Torchvision**
- **Matplotlib**
- **NumPy**

---

## 📊 Dataset MNIST
O dataset **MNIST** contém imagens de dígitos manuscritos (0–9) em escala de cinza:

- **Treino:** 60.000 imagens  
- **Teste/Validação:** 10.000 imagens  
- **Dimensão:** 28x28 pixels  

Exemplo de visualização de uma imagem:

```python
plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
plt.title(f"Label: {labels[0].item()}")
plt.show()
```

---

## 🏗️ Arquitetura da Rede Neural
A rede neural implementada possui três camadas **fully connected**:

```
Entrada (784 neurônios - pixels 28x28)
            │
            ▼
     [Camada Oculta 1]
        128 neurônios
            │
            ▼
     [Camada Oculta 2]
         64 neurônios
            │
            ▼
     [Camada de Saída]
        10 neurônios (dígitos 0–9)
```

- **Funções de ativação:**
  - ReLU nas camadas internas  
  - LogSoftmax na camada de saída  

---

## ⚙️ Treinamento
- **Otimizador:** `SGD` (Stochastic Gradient Descent) com `lr=0.01` e `momentum=0.5`  
- **Função de perda:** `NLLLoss`  
- **Épocas:** 10  

Durante o treinamento, a perda acumulada por época é exibida:

```text
Epoch 1 - Perda resultante: ...
Epoch 2 - Perda resultante: ...
...
Tempo de treino (em minutos) = ...
```

---

## ✅ Validação
A validação é realizada com o conjunto de teste.  
O modelo calcula a probabilidade de cada classe e compara com o rótulo correto.  
Ao final, é exibida a **precisão total**:

```text
Total de imagens testadas = 10000
Precisão do modelo = XX%
```

---

## 💻 Execução em GPU/CPU
O código detecta automaticamente se há GPU disponível (CUDA). Caso contrário, roda em CPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)
```

---

## 🚀 Como Executar
1. Clique no badge **Open in Colab** no topo ou acesse diretamente [este link](https://colab.research.google.com/drive/10Jb6rMh27h-tgDsE5Yx6iBM3oOMjsvlO?usp=sharing).  
2. Certifique-se de que o dataset MNIST será baixado automaticamente.  
3. Execute as células em ordem para treinar e validar o modelo.  

---

## 📈 Resultados Esperados
- Visualização de exemplos do dataset MNIST.  
- Treinamento da rede neural com redução progressiva da perda.  
- Validação com precisão aproximada entre **92% e 95%** (dependendo da configuração e número de épocas).  

---

## 🔮 Próximos Passos
- Implementar **camadas convolucionais (CNNs)** para melhorar a precisão.  
- Testar diferentes otimizadores como **Adam**.  
- Aumentar o número de épocas e ajustar hiperparâmetros.  
- Comparar desempenho entre CPU e GPU.  

---

## 📂 Estrutura do Projeto
```
├── Rede_neural_do_zero.ipynb   # Notebook principal
├── MNIST_data/                 # Dataset baixado automaticamente
└── README.md                   # Documentação do projeto
```

---

## 👨‍💻 Autor
Projeto desenvolvido como estudo prático de **Redes Neurais com PyTorch**.  

---

## 📝 Conclusão
Este estudo demonstrou, de forma prática e estruturada, como construir uma rede neural totalmente conectada (MLP) utilizando PyTorch para resolver o desafio clássico de classificação de dígitos manuscritos com o dataset MNIST.
A implementação percorreu todas as etapas fundamentais do ciclo de aprendizado de máquina:

• 	Pré-processamento dos dados, garantindo que as imagens fossem convertidas em tensores adequados para o modelo.

• 	Definição da arquitetura, com camadas lineares e funções de ativação que refletem os princípios básicos das redes neurais.

• 	Treinamento supervisionado, utilizando gradiente descendente estocástico para otimização dos parâmetros.

• 	Validação e avaliação, assegurando que o modelo fosse testado em dados não vistos e medindo sua capacidade de generalização.

Os resultados obtidos confirmam que, mesmo com uma arquitetura simples, é possível alcançar alta precisão (92–95%), evidenciando o potencial das redes neurais para tarefas de classificação.
Mais do que apenas treinar um modelo, este estudo reforça conceitos essenciais de aprendizado profundo, servindo como base sólida para avanços futuros, tais como:

• 	Implementação de redes convolucionais (CNNs) para maior desempenho em visão computacional.

• 	Exploração de diferentes otimizadores e hiperparâmetros para melhorar a eficiência do treinamento.

• 	Aplicações em datasets mais complexos e variados, ampliando o escopo e a robustez do modelo.

Em síntese, este projeto cumpre seu papel como um primeiro passo consistente na jornada de aprendizado em redes neurais, oferecendo uma visão clara e prática de como modelos de machine learning podem ser construídos, treinados e aplicados em problemas reais.
Ele não apenas consolida os fundamentos teóricos, mas também abre caminho para experimentações mais avançadas e aplicações em cenários de maior complexidade.


