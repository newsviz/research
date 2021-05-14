# research

Исследования и эксперименты для NewsViz Projects

[english version](#english-version)

## Структура репозитория

```bash
.
├── lemmatizers/ -- Лемматизация
│    ├── Comparing lemmatizers (standart dataset).ipynb -- сравнение лемматизаторов
│    └── Comparing lemmatizers(lenta_news).ipynb -- сравнение лемматизаторов на данных news_lenta
├── ner -- распознавание именованных сущностей
│    ├── md-extracting_NER.ipynb -- сравнение библиотек polyglot, natasha, spacy для NER
│    ├── vt-cleaned_names.ipynb -- извлечение имен
│    ├── vt-cleaned_names_rd.ipynb -- эксперименты с извлечением имен
│    └── extracting_NER.html -- .html версия сравнения библиотек для извлечение NER
└──topic_models -- тематическое моделирование
│    ├── bigartm
│        ├── bigARTM Baseline.ipynb -- базовое использование bigartm на данных news_lenta
│        ├── bigARTM_class.ipynb -- тематическое моделирование с модальностями
│        ├── bigARTM_hartm.ipynb -- иерархическая тематическая модель
│        ├── logging_functions.py
│        └── model.py -- класс для создания и обучения тематической модели 
│    ├── cor_ex
│        ├── tm_corextopic.ipynb -- базовое использование Correlation Explanation (CorEx)
│        └── visualizer
│            └── corex_topic
│                ├── pygal-corex_50-333.svg -- временная диаграмма с накоплением для тем
│                └── topic-model-example
│                    └── graphs
│                        ├── force.html
│                        ├── force.json
│                        ├── force_nontree.json
│                        ├── graph_prune_1000.dot
│                        ├── graph_prune_1000_sfdp.pdf
│                        ├── graph_prune_1000_sfdp_w_splines.pdf
│                        ├── tree.dot
│                        ├── tree_sfdp.pdf
│                        └── tree_sfdp_w_splines.pdf
│    ├── gensim
│        ├── Topic_model_lenta_time.ipynb -- Аанализ тем по времени для news_lenta
│        ├── Topic_modelling.ipynb -- базовое использование LDA
│        ├── Topic_modelling_more_topics_and_multicore.html -- .html версия Topic_modelling_more_topics_and_multicore.ipynb
│        ├── Topic_modelling_more_topics_and_multicore.ipynb -- базовое использование параллельной версии LDA
│        ├── Topic_modelling_ngram.ipynb -- LDA на биграммах и триграммах
│        ├── requirements.txt
│        └── tm_functions.py
│    ├── guided_lda
│        └── GuidedLDA.ipynb -- базовое использование Guided LDA
│    ├── sklearn
│        ├── nmf.ipynb -- тематическое моделирование с использованием неотрицательного матричного разложения
│        ├── tm_get_data.ipynb -- предобработка данных
│        ├── tm_sklearn.ipynb -- LDA из sklearn
│        └── visualizer
│            └── pyLDAvis_sklearn
│                ├── tsne_tf-dtm_sk.html -- intertopic Distance Map для CountVectorizer
│                └── tsne_tfidf-dtm_sk.html -- intertopic Distance Map для TfidfVectorizer
│    └── topsbm
│        ├── Dockerfile
│        └── TopSBM.ipynb -- тематическая модель на основе стохастических блочных моделей
```

# English version

All research and experiments in NewsViz Projects

## Repository structure

```bash
.
├── lemmatizers/ -- lemmatization
│    ├── Comparing lemmatizers (standart dataset).ipynb -- comparison of lemmatizers
│    └── Comparing lemmatizers(lenta_news).ipynb -- comparison of lemmatizers with news_lenta dataset
├── ner -- named-entity recognition
│    ├── md-extracting_NER.ipynb -- comparison of polyglot, natasha, spacy for NER
│    ├── vt-cleaned_names.ipynb -- name extraction
│    ├── vt-cleaned_names_rd.ipynb -- r&d with a name extraction
│    └── extracting_NER.html -- .html version for md-extracting_NER.ipynb
└──topic_models -- topic modeling
│    ├── bigartm
│        ├── bigARTM Baseline.ipynb -- bigartm baseline with news_lenta dataset
│        ├── bigARTM_class.ipynb -- topic modeling with class
│        ├── bigARTM_hartm.ipynb -- topic modeling with a hierarchy 
│        ├── logging_functions.py
│        └── model.py -- the class for topic model training
│    ├── cor_ex
│        ├── tm_corextopic.ipynb -- Correlation Explanation (CorEx) baseline
│        └── visualizer
│            └── corex_topic
│                ├── pygal-corex_50-333.svg -- stacked timing chart for topics
│                └── topic-model-example
│                    └── graphs
│                        ├── force.html
│                        ├── force.json
│                        ├── force_nontree.json
│                        ├── graph_prune_1000.dot
│                        ├── graph_prune_1000_sfdp.pdf
│                        ├── graph_prune_1000_sfdp_w_splines.pdf
│                        ├── tree.dot
│                        ├── tree_sfdp.pdf
│                        └── tree_sfdp_w_splines.pdf
│    ├── gensim
│        ├── Topic_model_lenta_time.ipynb -- topic analysis by time for news_lenta dataset
│        ├── Topic_modelling.ipynb -- LDA baseline
│        ├── Topic_modelling_more_topics_and_multicore.html -- .html version of Topic_modelling_more_topics_and_multicore.ipynb
│        ├── Topic_modelling_more_topics_and_multicore.ipynb -- multicore LDA baseline
│        ├── Topic_modelling_ngram.ipynb -- LDA with bigrams and trigrams
│        ├── requirements.txt
│        └── tm_functions.py
│    ├── guided_lda
│        └── GuidedLDA.ipynb -- Guided LDA baseline
│    ├── sklearn
│        ├── nmf.ipynb -- topic modeling with non-negative matrix factorization
│        ├── tm_get_data.ipynb -- data preprocessing
│        ├── tm_sklearn.ipynb -- LDA from sklearn package
│        └── visualizer
│            └── pyLDAvis_sklearn
│                ├── tsne_tf-dtm_sk.html -- Intertopic Distance Map для CountVectorizer
│                └── tsne_tfidf-dtm_sk.html -- Intertopic Distance Map для TfidfVectorizer
│    └── topsbm
│        ├── Dockerfile
│        └── TopSBM.ipynb -- topic model based on Stochastic Block Models
```
