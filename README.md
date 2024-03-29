# Описание
Это веб-приложение позволяет загружать текстовые файлы и анализировать их содержание, вычисляя TF (Term Frequency) и IDF (Inverse Document Frequency) для каждого слова в тексте. После обработки файла отображается таблица с 50 наиболее значимыми словами, упорядоченными по убыванию IDF.

# Требования
- Docker

# Запуск
```sh
git clone https://github.com/demig00d/tfidf-web-app.git
```

```sh
cd tfidf-web-app
```

```sh
docker build . -t tfidf-web-app
```

```sh
docker run -p 8000:8000 tfidf-web-app
```
