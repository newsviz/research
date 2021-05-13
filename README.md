# research
All research and experiments in NewsViz Projects

## Структура репозитория

```bash
.
├── lemmatizers/ -- Лемматизация
│    ├── Comparing lemmatizers (standart dataset).ipynb -- сравнение лемматизаторов
│    └── Comparing lemmatizers(lenta_news).ipynb -- сравнение лемматизаторов на данных lenta_news
├── ner -- Распознавание именованных сущностей
│    ├── md-extracting_NER.ipynb -- Сравнение библиотек polyglot, natasha, spacy для извлечение NER
│    ├── vt-cleaned_names.ipynb -- Извлечение имен
│    ├── vt-cleaned_names_rd.ipynb -- Эксперименты с извлечением имен
│    └── extracting_NER.html -- 
└──topic_models -- Тематическое моделирование
│    ├── bigartm
│        ├── bigARTM Baseline.ipynb
│        ├── bigARTM_class.ipynb
│        ├── bigARTM_hartm.ipynb
│        ├── logging_functions.py
│        └── model.py
│    ├── cor_ex
│        ├── tm_corextopic.ipynb
│        └── visualizer
│            └── corex_topic
│                ├── pygal-corex_50-333.svg
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
│        ├── Topic_model_lenta_time.ipynb
│        ├── Topic_modelling.ipynb
│        ├── Topic_modelling_more_topics_and_multicore.html
│        ├── Topic_modelling_more_topics_and_multicore.ipynb
│        ├── Topic_modelling_ngram.ipynb
│        ├── requirements.txt
│        └── tm_functions.py
│    ├── guided_lda
│        └── GuidedLDA.ipynb
│    ├── sklearn
│        ├── nmf.ipynb
│        ├── tm_get_data.ipynb
│        ├── tm_sklearn.ipynb
│        └── visualizer
│            └── pyLDAvis_sklearn
│                ├── tsne_tf-dtm_sk.html
│                └── tsne_tfidf-dtm_sk.html
│    └── topsbm
│        ├── Dockerfile
│        └── TopSBM.ipynb
```
