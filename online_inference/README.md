### Usage Instruction
~~~
Basic usage steps:

    cd online_inference
    docker build -t averpower/homework2 .
    docker run -p 8000:8000 averpower/homework2
    python make_request.py {num}
        num - number of requests
~~~

#### Other Options
~~~
You can pull docker image from https://hub.docker.com/:

    docker pull averpower/homework2:latest
    docker run -p 8000:8000 averpower/homework2
~~~

Self-rating of homework #2

| Task Description   | Score |
| ------------- | ------------- |
| Оберните inference вашей модели в rest сервис | 3 |
| Напишите тест для /predict  | 3  |
| Напишите скрипт, который будет делать запросы к вашему сервису  | 2 |
| Сделайте валидацию входных данных  | 3  |   
| Напишите dockerfile | 4  |
| Оптимизируйте размер docker image  | 0 |   
| Опубликуйте образ в https://hub.docker.com/  | 2  |
| напишите в readme корректные команды docker pull/run  | 1  |   
| Проведите самооценку | 1  |
|   |   |   
| Итого  |  19 |   