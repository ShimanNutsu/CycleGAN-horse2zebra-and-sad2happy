Была решена задачи релизации архитектуры Cycle GAN и применение модели для задачи преобразования коня в зебру и наоборот (задача, решенная в статье) и преобразования грустного лица в веселое и наоборот (своя задача).

В файле models.py представлены модели генератора и дискриминатора.
Архитектуры моделей были взяты из статьи, возможно с малыми изменениями (на фото). Цикл обучения применён обычный (без буфера сгенерированных изображений)
![image](https://user-images.githubusercontent.com/46298358/153759725-3037182a-153f-4947-9c71-5fbb3a19dfea.png)

Результаты работы моделей:
![image](https://user-images.githubusercontent.com/46298358/153759857-d8b7fb65-1a6b-4700-9ab2-c5f4f57198ed.png)
![image](https://user-images.githubusercontent.com/46298358/153759863-e9de0a22-57f7-4952-8b8b-dbd534436f24.png)
![image](https://user-images.githubusercontent.com/46298358/153759873-f89851ff-d74f-417b-87d3-78c7b7540d06.png)
![image](https://user-images.githubusercontent.com/46298358/153759879-a6abf4fd-3272-4955-b3ad-ee3db727517b.png)
