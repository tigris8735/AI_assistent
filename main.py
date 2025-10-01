import numpy as np
import random
import json
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn

# Скачиваем необходимые данные NLTK
nltk.download('punkt')

class ProgrammingAssistant:
    def __init__(self):
        self.name = "Маширо"
        self.stemmer = PorterStemmer()
        
        # База знаний с паттернами вопросов и ответов
        self.intents = {
            "greeting": {
                "patterns": ["привет", "здравствуй", "добрый день", "хай"],
                "responses": ["Привет! Я Маширо, ваш ассистент по программированию.", "Здравствуйте! Готов помочь с кодом."]
            },
            "ml_definition": {
                "patterns": ["что такое машинное обучение", "объясни ml", "как работает ml"],
                "responses": ["Машинное обучение - это область ИИ, где компьютеры обучаются на данных без явного программирования."]
            },
            "python_help": {
                "patterns": ["помоги с python", "проблема с питоном", "как написать код на python"],
                "responses": ["Для Python рекомендую использовать документацию и библиотеки like NumPy для ML."]
            },
            "debug": {
                "patterns": ["ошибка в коде", "помоги исправить код", "дебаг"],
                "responses": ["Опишите ошибку подробнее, попробуем разобраться вместе."]
            }
        }
        
        # Собираем все фразы для обучения
        self.patterns = []
        self.tags = []
        for intent_name, intent_data in self.intents.items():
            for pattern in intent_data['patterns']:
                self.patterns.append(self.preprocess_text(pattern))
                self.tags.append(intent_name)
        
        # Векторизатор текста
        self.vectorizer = TfidfVectorizer()
        if self.patterns:
            self.X = self.vectorizer.fit_transform(self.patterns)
        
        # Простая нейросеть для демонстрации
        self.model = self.create_model()

    def preprocess_text(self, text):
        """Предобработка текста"""
        tokens = nltk.word_tokenize(text.lower())
        stemmed = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed)

    def create_model(self):
        """Создаем простую нейронную сеть"""
        class NeuralNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(NeuralNet, self).__init__()
                self.l1 = nn.Linear(input_size, hidden_size)
                self.l2 = nn.Linear(hidden_size, hidden_size)
                self.l3 = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                out = self.l1(x)
                out = self.relu(out)
                out = self.l2(out)
                out = self.relu(out)
                out = self.l3(out)
                return out
        
        return NeuralNet(100, 64, len(set(self.tags))) if self.patterns else None

    def get_response(self, user_input):
        """Получить ответ от ассистента"""
        processed_input = self.preprocess_text(user_input)
        
        # Поиск похожего вопроса с помощью косинусной схожести
        input_vec = self.vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vec, self.X)
        
        if np.max(similarities) > 0.5:
            best_match_idx = np.argmax(similarities)
            tag = self.tags[best_match_idx]
            return random.choice(self.intents[tag]['responses'])
        else:
            return "Извините, я еще учусь. Можете переформулировать вопрос?"

    def train_ml_model(self):
        """Демонстрация тренировки ML модели"""
        print(f"{self.name}: Тренирую модель машинного обучения...")
        # Здесь может быть реальная тренировка модели
        print(f"{self.name}: Модель готова к работе!")

# Главный цикл ассистента
def main():
    assistant = ProgrammingAssistant()
    assistant.train_ml_model()
    
    print(f"{assistant.name}: Привет! Я ваш ассистент по программированию с ML. Задавайте вопросы!")
    
    while True:
        user_input = input("Вы: ")
        
        if user_input.lower() in ['выход', 'exit', 'quit']:
            print(f"{assistant.name}: До свидания! Удачи в программировании!")
            break
        
        response = assistant.get_response(user_input)
        print(f"{assistant.name}: {response}")

if __name__ == "__main__":
    main()