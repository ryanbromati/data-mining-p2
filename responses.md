# Análise de Anomalias em Dados de Passageiros

## O Que São Anomalias?
Anomalias são padrões nos dados que não seguem o comportamento esperado. Em análise de dados, identificamos esses "casos atípicos" para entender perfis diferentes dos usuários comuns.

## Grupo Incomum de Passageiros Encontrado

Encontramos um grupo especial de passageiros (chamado Cluster 1) que representa **6,19%** do total de passageiros do aeroporto (174 pessoas).

### Características Deste Grupo:

- **Satisfação muito alta**: Pontuaram 9,75 em satisfação, bem acima da média geral
- **Muito jovens**: Idade bem abaixo da média dos outros passageiros
- **Predominância de um gênero**: Apresentam forte tendência para um gênero específico
- **Baixa renda**: Renda significativamente menor que a média
- **Voam raramente**: Frequência de voos muito menor que a média
- **Experiência de voo normal**: Experiência de voos levemente acima da média
- **Tempo usando o aeroporto normal**: Tempo de uso do aeroporto próximo à média

Em resumo, este grupo representa passageiros jovens, de baixa renda, que não viajam frequentemente, mas que, curiosamente, estão muito mais satisfeitos com o serviço do aeroporto que os demais grupos.

---

# Padrões de Leitura em Portal de Notícias

## O Que São Regras de Associação?
Regras de associação são padrões do tipo "se isso, então aquilo". Por exemplo, se uma pessoa lê notícias sobre tecnologia e ciência, ela provavelmente também se interessa por educação. Estas regras ajudam a prever comportamentos e fazer recomendações.

## O Problema Estudado

Identificar padrões nos interesses de leitura dos usuários de um portal de notícias para poder recomendar conteúdos relevantes baseados nas categorias que eles já acessaram.

## Dados Analisados

- **225.230 registros** de navegação no site
- **50.000 sessões** de usuários diferentes
- **43.838 sessões válidas** após limpeza dos dados
- **15 categorias** de notícias: Saúde, Viagem, Gastronomia, Lifestyle, Entretenimento, Cultura, Negócios, Política, Ciência, Tecnologia, Opinião, Educação, Internacional, Economia, Esportes

## Como Foi Feita a Análise

1. **Preparação dos dados**: Organizamos os dados para mostrar quais categorias cada usuário acessou em uma sessão

2. **Algoritmo FP-Growth**: Usamos este algoritmo (mais eficiente que o tradicional Apriori) para encontrar combinações frequentes de categorias

3. **Medidas importantes**:
   - **Suporte**: Frequência com que categorias aparecem juntas (quanto maior, mais comum)
   - **Confiança**: Probabilidade de um usuário acessar Y quando já acessou X
   - **Lift**: Mede o quanto as categorias estão realmente relacionadas além do acaso (acima de 1 indica boa relação)

## Resultados Encontrados

A análise encontrou padrões interessantes, como por exemplo: quem lê sobre Lifestyle e Gastronomia tem alta probabilidade de também se interessar por Viagem.

Estes padrões permitem criar sistemas de recomendação que sugerem notícias com maior chance de interesse para cada usuário.

---

# Padrões de Matrícula em Cursos Universitários

## O Problema Estudado

Descobrir quais disciplinas universitárias os alunos tendem a cursar juntas. Estes padrões podem ajudar em:

- Recomendação de cursos para os estudantes
- Melhor planejamento de horários das aulas
- Previsão da demanda por cada disciplina
- Identificação de relações entre disciplinas

## Dados Analisados

Usamos dados simulados de:
- **1.000 estudantes**
- **20 disciplinas** em 8 departamentos diferentes
- Cada estudante matriculado em **4 a 8 disciplinas**
- Algumas disciplinas têm pré-requisitos ou são naturalmente cursadas juntas

## Como Foi Feita a Análise

1. **Preparação dos dados**: Organizamos os dados mostrando quais disciplinas cada estudante cursou

2. **Identificação de padrões frequentes**: Usamos o algoritmo FP-Growth para encontrar conjuntos de disciplinas que frequentemente são cursadas juntas (em pelo menos 5% dos casos)

3. **Geração das regras**: Criamos regras de associação com confiança mínima de 70% (ou seja, a regra precisa estar correta em pelo menos 70% das vezes)

4. **Avaliação**: Analisamos os resultados usando medidas como suporte, confiança e lift

## Descobertas Principais

Identificamos 5 regras importantes:

1. **CS101 → CS201, MATH101** (Confiança: 81%, Lift: 3,2)
   *Tradução: Estudantes que fazem Introdução à Computação geralmente também cursam Programação Avançada e Cálculo I*

2. **MATH101 → PHYS101** (Confiança: 75%, Lift: 2,9)
   *Tradução: Alunos de Cálculo I frequentemente também se matriculam em Física I*

3. **CHEM101 → BIO101** (Confiança: 72%, Lift: 2,8)
   *Tradução: Existe uma forte relação entre Química Geral e Biologia Geral*

4. **CS201 → CS301** (Confiança: 89%, Lift: 3,5)
   *Tradução: Alunos de Programação Avançada quase sempre progridem para Estruturas de Dados*

5. **MATH201 → MATH301** (Confiança: 77%, Lift: 3,1)
   *Tradução: Estudantes de Cálculo II geralmente continuam para Cálculo III*

## Como Usar Estes Resultados

Estas descobertas podem ser usadas para:

- **Orientação acadêmica**: Sugerir disciplinas complementares aos estudantes
- **Melhor programação de horários**: Evitar conflitos de horário entre disciplinas frequentemente cursadas juntas
- **Aprimoramento do currículo**: Adaptar os currículos aos caminhos naturais que os estudantes seguem
- **Identificação de conexões entre disciplinas**: Descobrir dependências de conhecimento não formalizadas
- **Previsão de demanda**: Antecipar a procura por disciplinas específicas

Este exemplo mostra como técnicas normalmente usadas em análise de compras podem ser aplicadas com sucesso na educação, melhorando a experiência dos alunos e ajudando no planejamento institucional.