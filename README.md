# research
All research and experiments in NewsViz Projects

## Структура репозитория

```bash
.
├── lemmatizers/ -- Лемматизация
│    ├── Comparing lemmatizers (standart dataset).ipynb -- сравнение лемматизаторов
│    └── Comparing lemmatizers(lenta_news).ipynb -- сравнение лемматизаторов на данных news_lenta
├── ner -- Распознавание именованных сущностей
│    ├── md-extracting_NER.ipynb -- Сравнение библиотек polyglot, natasha, spacy для извлечение NER
│    ├── vt-cleaned_names.ipynb -- Извлечение имен
│    ├── vt-cleaned_names_rd.ipynb -- Эксперименты с извлечением имен
│    └── extracting_NER.html -- .html версия сравнения библиотек для извлечение NER
└──topic_models -- Тематическое моделирование
│    ├── bigartm
│        ├── bigARTM Baseline.ipynb -- Базовое использование bigartm на данных news_lenta
│        ├── bigARTM_class.ipynb -- Тематическое моделирование с модальностями
│        ├── bigARTM_hartm.ipynb -- Иерархическая тематическая модель
│        ├── logging_functions.py
│        └── model.py -- Класс для создания и обучения тематической модели 
│    ├── cor_ex
│        ├── tm_corextopic.ipynb -- Базовое использование Correlation Explanation (CorEx)
│        └── visualizer
│            └── corex_topic
│                ├── pygal-corex_50-333.svg -- Временная диаграмма с накоплением для тем
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
│        ├── Topic_model_lenta_time.ipynb -- Анализ тем по времени для news_lenta
│        ├── Topic_modelling.ipynb -- Базовое использование LDA
│        ├── Topic_modelling_more_topics_and_multicore.html --
│        ├── Topic_modelling_more_topics_and_multicore.ipynb -- Базовое использование параллельной версии LDA
│        ├── Topic_modelling_ngram.ipynb -- LDA на биграммах и триграммах
│        ├── requirements.txt
│        └── tm_functions.py
│    ├── guided_lda
│        └── GuidedLDA.ipynb -- Базовое использование Guided LDA
│    ├── sklearn
│        ├── nmf.ipynb -- Тематическое моделирование с использованием неотрицательного матричного разложения
│        ├── tm_get_data.ipynb -- Предобоработка данных
│        ├── tm_sklearn.ipynb -- LDA из sklearn
│        └── visualizer
│            └── pyLDAvis_sklearn
│                ├── tsne_tf-dtm_sk.html -- Intertopic Distance Map для CountVectorizer
│                └── tsne_tfidf-dtm_sk.html -- Intertopic Distance Map для TfidfVectorizer
│    └── topsbm
│        ├── Dockerfile
│        └── TopSBM.ipynb -- Тематическая модель на основе стохастических блочных моделей
```
