# AVSS AUNet Framework

Небольшая инструкция, как работать с нашим фреймворком.

Как запустить обучение:
1) Нужно выбрать модель
2) Нужно выбрать тип обучения: полное или onebatchtest
3) Выбрать нужный конфиг
4) На всякий случай нужно проверить пути до данных, сейчас есть два конфига для датасета: путь до приватного датасета на kaggle и путь на виртуальной машине к датасета, но скорее всего он не подойдет. Можно в любой из конфигах прописать свой путь до датасета
5) Вставить название конфига модели в `train.py`
6) И в консоли прописать `python3 path/to/train.py`


Как запустить инференс:
1) Нужно скачать веса модели. Для этого есть скрипт `download_weights.py`, который скачивает веса модели с гугл диска. Есть конфиг `download.yaml`, где указывается ссылка на гугл диск с весами и название директории, куда сохранится модель. По умолчанию указана ссылка на веса модели ConvTasNet и название директории - **best_conv_tasnet**. Полный путь, куда сохраняется модель: `path/to/download_weights.py/../saved/dir_name_from_download.yaml/best_model.pth`. Запустить скрипт: `python3 path/to/download_weights.py`
2) в конфиге `inference.yaml` указать путь до pretrained модели и прописать путь до датасета: путь до директорий mix, s1 и s2 соответственно.
3) Запустить: `python3 path/to/inference.py`


Как запустить расчет метрик:
1) Есть скрипт `calculate_metrics.py`. Есть конфиг `metric.yaml`, куда нужно указать пути до директорий: с s1_estimated, s2_estimated, s1_target, s2_target и mix.
2) Запустить: `python3 path/to/calculate_metrics.py` и в консоли появятся метрики