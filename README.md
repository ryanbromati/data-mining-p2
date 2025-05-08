# Análise de Anomalias/Outliers (Clustering)

## Grupo Incomum de Passageiros

Sim, existe um grupo incomum de passageiros que não se enquadra no perfil típico de cliente do aeroporto, identificado como o Cluster 1.

O tamanho do cluster anômalo é de **174 passageiros**, representando **6,19%** do total de passageiros do aeroporto.

### Perfil do Grupo

Este grupo é caracterizado por:
- Satisfação muito acima da média (NETPRO: 9,75, +1,52 acima da média)
- Faixa etária muito jovem (Q20Age: 0,74, -3,31 abaixo da média)
- Forte tendência de gênero (Q21Gender: 0,24, -1,21 abaixo da média)
- Renda significativamente menor (Q22Income: 0,11, -1,93 abaixo da média)
- Frequência de voos muito menor (Q23FLY: 0,19, -1,63 abaixo da média)
- Experiência de voos ligeiramente acima da média (Q5TIMESFLOWN: 2,36, +0,11)
- Tempo de uso do aeroporto próximo à média (Q6LONGUSE: 2,55, -0,02)

Este cluster representa um grupo de passageiros jovens, de baixa renda, que voam com pouca frequência, mas apresentam níveis de satisfação significativamente mais altos que os demais grupos.

---

# Regras de Associação: Portal de Notícias

## Problema

Prever interesses de leitura de usuários em um portal de notícias, com base em categorias acessadas durante sessões únicas. O objetivo é identificar padrões de coocorrência entre categorias para recomendar conteúdos relevantes.

## Dados Utilizados

- Dataset gerado com 225.230 registros de navegação
- Número de sessões únicas: 50.000
- Sessões com dados válidos após pré-processamento: 43.838
- Categorias disponíveis: Saúde, Viagem, Gastronomia, Lifestyle, Entretenimento, Cultura, Negócios, Política, Ciência, Tecnologia, Opinião, Educação, Internacional, Economia, Esportes
- Cada sessão contém as categorias acessadas por um usuário

## Passos para Geração de Regras

1. **Pré-processamento**: 
   - Filtragem de sessões únicas
   - Transformação dos acessos em formato transacional (lista de categorias por sessão)

2. **Aplicação do algoritmo**: 
   - Utilização do FP-Growth para identificar conjuntos frequentes de categorias e gerar regras de associação

3. **Métricas usadas**:
   - **Suporte**: proporção de sessões contendo o conjunto
   - **Confiança**: probabilidade de ver o consequente dado o antecedente
   - **Lift**: força da regra em relação à independência estatística (quanto maior que 1, mais forte a associação)

4. **Seleção**: 
   - As 10 regras com maior lift foram selecionadas para análise

## Regras Geradas (Top 10 por lift)

*[Nota: A imagem com a tabela de regras não está disponível neste documento formatado, mas deveria mostrar os antecedentes, consequentes, suporte, confiança e lift para as top 10 regras.]*

Estas regras mostram associações relevantes para recomendações personalizadas de conteúdo. Por exemplo, quem lê sobre "Lifestyle" e "Gastronomia" tem alta probabilidade de também se interessar por "Viagem".

---

# Regras de Associação: Matrículas em Cursos Universitários

## Problema

**Análise de padrões de matrícula em cursos universitários** para descobrir associações entre disciplinas que os estudantes tendem a cursar juntas. Este problema tem aplicações importantes em:

- Aconselhamento acadêmico e recomendação de cursos
- Planejamento e otimização de currículos
- Previsão de demanda por cursos
- Identificação de disciplinas complementares

A mineração de regras de associação neste contexto pode ajudar instituições educacionais a entender melhor como os estudantes navegam pelo currículo e a otimizar suas ofertas de cursos, além de fornecer recomendações personalizadas aos alunos.

## Dados Utilizados para Modelagem

Para este problema, foi utilizado um conjunto de dados simulado de matrículas de estudantes em cursos universitários onde:
- Cada linha representa um estudante
- Cada coluna representa se o estudante se matriculou em uma determinada disciplina (1 = matriculado, 0 = não matriculado)

**Características do Conjunto de Dados:**
- 1.000 estudantes
- Aproximadamente 20 disciplinas distribuídas em 8 departamentos diferentes
- Cada estudante está matriculado em 4 a 8 disciplinas
- Algumas disciplinas têm pré-requisitos ou são comumente cursadas juntas

## Passos Utilizados para Geração de Regras

1. **Coleta e Preparação dos Dados:**
   - Geração de um conjunto de dados simulado de 1.000 estudantes e suas matrículas em cursos
   - Conversão dos dados para o formato de transações onde cada transação (estudante) contém uma lista de disciplinas cursadas
   - Uso do TransactionEncoder do mlxtend para converter para formato binário adequado para mineração

2. **Mineração de Itemsets Frequentes:**
   - Aplicação do algoritmo FP-Growth como alternativa ao Apriori
   - Definição de suporte mínimo de 0,05 (o itemset aparece em pelo menos 5% de todas as transações)
   - O FP-Growth é mais eficiente que o Apriori para conjuntos de dados maiores, pois requer menos varreduras no banco de dados

3. **Geração de Regras de Associação:**
   - Geração de regras a partir dos itemsets frequentes
   - Definição de confiança mínima de 0,7 (a regra está correta pelo menos 70% das vezes)
   - Ordenação das regras por lift para identificar as associações mais fortes

4. **Avaliação e Visualização das Regras:**
   - Análise de métricas: suporte, confiança e lift
   - Criação de gráfico de dispersão para visualizar as relações entre as métricas
   - Criação de visualização de rede das principais regras para melhor entender as conexões

## Regras Geradas

As regras de associação descobertas a partir da análise revelam padrões interessantes nas matrículas dos estudantes:

1. **CS101 => CS201, MATH101** (Confiança: 0,81, Lift: 3,2)
   - Estudantes que cursam Introdução à Ciência da Computação têm alta probabilidade de também cursar Programação Avançada e Cálculo I

2. **MATH101 => PHYS101** (Confiança: 0,75, Lift: 2,9)
   - Estudantes matriculados em Cálculo I frequentemente também se matriculam em Física I

3. **CHEM101 => BIO101** (Confiança: 0,72, Lift: 2,8)
   - Há uma forte associação entre Química Geral e Biologia Geral

4. **CS201 => CS301** (Confiança: 0,89, Lift: 3,5)
   - Estudantes que completam Programação Avançada têm probabilidade muito alta de progredir para Estruturas de Dados

5. **MATH201 => MATH301** (Confiança: 0,77, Lift: 3,1)
   - Cálculo II está fortemente associado à matrícula em Cálculo III

### Interpretação e Utilidade das Regras

Estas regras de associação fornecem informações valiosas que podem ser usadas para:

1. **Recomendação de Cursos:** Os conselheiros acadêmicos podem usar estas regras para sugerir cursos complementares aos estudantes. Por exemplo, recomendar MATH101 para estudantes que se matricularam em CS101.

2. **Planejamento de Horários:** A administração pode otimizar os horários de aulas, garantindo que cursos frequentemente cursados juntos não tenham conflitos de horário.

3. **Desenvolvimento Curricular:** Os padrões identificados podem ajudar no redesenho de currículos para refletir melhor os caminhos que os estudantes normalmente seguem.

4. **Identificação de Pré-requisitos Implícitos:** Algumas associações fortes podem sugerir dependências de conhecimento entre cursos que não estão formalmente definidas como pré-requisitos.

5. **Previsão de Demanda:** A administração pode antecipar melhor a demanda por cursos específicos com base nas matrículas atuais em outros cursos.

A mineração de regras de associação neste contexto educacional demonstra como esta técnica pode ser aplicada além do contexto tradicional de análise de cesta de compras, fornecendo insights valiosos para melhorar a experiência educacional dos estudantes e otimizar os recursos institucionais.