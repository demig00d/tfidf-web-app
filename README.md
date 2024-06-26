# Описание
Это веб-приложение предназначено для загрузки и анализа текстового файлов. Приложение вычисляет TF (Term Frequency) и IDF (Inverse Document Frequency) для каждого слова в тексте, затем выводит результат в виде таблицы.

> [!NOTE]
> По условиям ([ссылка](https://gist.github.com/nonamenix/651852a8943e6a84abdf03ed82dc7518#%D1%82%D0%B5%D1%81%D1%82%D0%BE%D0%B2%D0%BE%D0%B5-%D0%B7%D0%B0%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5)) TF и IDF считаются для одного текстового файла, поэтому IDF - всегда будет равен 1

# Требования
- Docker

# Запуск
1. Клонировать репозиторий:
```sh
git clone https://github.com/demig00d/tfidf-web-app.git
```
2. Перейти в директорию проекта:
```sh
cd tfidf-web-app
```
3. Собрать Docker образ:
```sh
docker build . -t tfidf-web-app
```
4. Запустить Docker контейнер:
```sh
docker run -p 8000:8000 tfidf-web-app
```
5. Открыть http://localhost:8000

# Возможные улучшения
- кэширование результатов
- очередь задач
- читать и обрабатывать большие файлы с помощью генераторов кусками
