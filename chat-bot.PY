import os
import re
import ast
import operator
import json
import random
from datetime import datetime
import numpy as np

# Файл с именем лежит рядом со скриптом, поэтому имя помнится и между запусками
NAME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_memory.json")

# Намерения: примеры фраз и ответы. Добавляй свои!
INTENTS = {
    "приветствие": {
        "patterns": ["привет", "здравствуй", "добрый день", "доброе утро",
                     "добрый вечер", "хай", "здорово", "приветствую"],
        "responses": ["Привет! Чем могу помочь?", "Здравствуй! Рад тебя видеть."],
    },
    "прощание": {
        "patterns": ["пока", "до свидания", "увидимся", "всего хорошего",
                     "я пошёл", "до встречи", "прощай"],
        "responses": ["Пока! Заходи ещё.", "До встречи!"],
    },
    "дела": {
        "patterns": ["как дела", "как ты", "как жизнь", "как поживаешь",
                     "что нового", "как настроение"],
        "responses": ["Отлично, я же программа :) А у тебя как?",
                      "Всё хорошо, спасибо!"],
    },
    "имя": {
        "patterns": ["как тебя зовут", "кто ты", "твоё имя", "ты кто",
                     "представься", "как твоё имя"],
        "responses": ["Я простой чат-бот на numpy.", "Я бот, написанный на numpy."],
    },
    "спасибо": {
        "patterns": ["спасибо", "благодарю", "спс", "большое спасибо", "thanks"],
        "responses": ["Пожалуйста!", "Рад помочь!"],
    },
    "время": {
        "patterns": ["который час", "сколько времени", "время", "скажи время",
                     "сколько сейчас времени"],
        "responses": [],  # обрабатывается отдельно
    },
    "шутка": {
        "patterns": ["расскажи шутку", "пошути", "расскажи анекдот", "анекдот",
                     "хочу шутку", "рассмеши меня", "шутку давай", "повесели меня",
                     "расскажи что-нибудь смешное", "шутка"],
        "responses": [
            "Программист ложится спать. Жена: «Закрой окно, холодно!» — «И что, если я закрою окно, станет тепло?»",
            "Почему программисты путают Хэллоуин и Рождество? Потому что Oct 31 == Dec 25.",
            "Заходит нейросеть в бар. Бармен: «Вам как обычно?» Нейросеть: «Мне нужно ещё 10 000 примеров».",
            "— Сколько программистов нужно, чтобы поменять лампочку? — Нисколько, это аппаратная проблема.",
            "Баг — это не ошибка, а недокументированная фича.",
            "numpy сказал массиву: «Не переживай, я тебя растяну» — это был бродкастинг.",
        ],
    },
    "моё_имя": {
        "patterns": ["как меня зовут", "ты знаешь моё имя", "помнишь моё имя",
                     "знаешь как меня зовут", "скажи моё имя", "кто я"],
        "responses": [],  # обрабатывается отдельно
    },
    "умения": {
        "patterns": ["что ты умеешь", "чем можешь помочь", "твои возможности",
                     "помощь", "что можешь"],
        "responses": ["Я умею здороваться, прощаться, болтать о делах, говорить время, "
                      "запоминать твоё имя, шутить и считать (например: «посчитай 2+3*4»)."],
    },
}


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}


def _safe_eval(node):
    """Безопасное вычисление: разрешены только числа и арифметика, никакого eval()."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _safe_eval(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 100:
            raise ValueError("слишком большая степень")
        return _OPS[type(node.op)](a, b)
    raise ValueError("недопустимое выражение")


def calculate(text):
    """Возвращает ответ-строку, если в тексте есть выражение, иначе None."""
    t = text.lower()
    # «10 в -3 степени» -> «10**(-3)»
    t = re.sub(r"\s+в\s+(-?\d+(?:[.,]\d+)?)\s+степени", r"**(\1)", t)
    for word, sym in [("умножить на", "*"), ("разделить на", "/"), ("делить на", "/"),
                      ("в степени", "**"), ("плюс", "+"), ("минус", "-"),
                      ("умножить", "*"), ("^", "**"), ("×", "*"), ("÷", "/"),
                      ("x", "*"), ("х", "*"), (":", "/"), (",", ".")]:
        t = t.replace(word, sym)
    m = re.search(r"[\d(][\d\s+\-*/().%]*[\d)]", t)
    if not m or not re.search(r"[+\-*/%]", m.group()):
        return None
    expr = m.group().strip()
    try:
        result = _safe_eval(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return "На ноль делить нельзя!"
    except (ValueError, SyntaxError, OverflowError):
        return "Не смог посчитать. Напиши, например: «посчитай 2+3*4»."
    if isinstance(result, float):
        if result == int(result):
            result = int(result)
        else:
            # обычная запись вместо 1e-05
            result = f"{result:.8f}".rstrip("0").rstrip(".")
    if isinstance(result, int) and len(str(result)) > 30:
        result = f"{result:.4e}"  # очень длинные числа — в научной записи
    return f"{expr} = {result}"


def tokenize(text):
    # берём первые 5 букв слова — простой «стемминг» для русского языка
    return [w[:5] for w in re.findall(r"[а-яёa-z0-9]+", text.lower())]


class ChatBot:
    def __init__(self, intents, hidden=16, lr=0.5, epochs=400, seed=0):
        self.intents = intents
        self.last_tag = None
        self.name = self.load_name()
        self.tags = list(intents)
        words = {w for d in intents.values() for p in d["patterns"] for w in tokenize(p)}
        self.vocab = {w: i for i, w in enumerate(sorted(words))}

        X, y = [], []
        for ti, tag in enumerate(self.tags):
            for p in intents[tag]["patterns"]:
                X.append(self.vectorize(p))
                y.append(ti)
        X = np.array(X)
        Y = np.eye(len(self.tags))[y]

        rng = np.random.default_rng(seed)
        V, C = X.shape[1], len(self.tags)
        self.W1 = rng.standard_normal((V, hidden)) * 0.5
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, C)) * 0.5
        self.b2 = np.zeros(C)

        for _ in range(epochs):
            h, p = self.forward(X)
            d2 = (p - Y) / len(X)
            d1 = (d2 @ self.W2.T) * (h > 0)
            self.W2 -= lr * h.T @ d2
            self.b2 -= lr * d2.sum(axis=0)
            self.W1 -= lr * X.T @ d1
            self.b1 -= lr * d1.sum(axis=0)

    def vectorize(self, text):
        v = np.zeros(len(self.vocab))
        for w in tokenize(text):
            if w in self.vocab:
                v[self.vocab[w]] = 1
        return v

    @staticmethod
    def softmax(z):
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)  # ReLU
        return h, self.softmax(h @ self.W2 + self.b2)

    def load_name(self):
        try:
            with open(NAME_FILE, encoding="utf-8") as f:
                return json.load(f).get("name")
        except (OSError, ValueError):
            return None

    def save_name(self, name):
        self.name = name
        try:
            with open(NAME_FILE, "w", encoding="utf-8") as f:
                json.dump({"name": name}, f, ensure_ascii=False)
        except OSError:
            pass  # если записать нельзя, имя живёт только до конца сеанса

    def reply(self, text):
        # Имя ищем по шаблону, до нейросети: «меня зовут Максим», «зови меня Макс»
        m = re.search(r"(?:меня зовут|моё имя|мое имя|зови меня)\s+([а-яёa-z-]+)",
                      text.lower())
        if m:
            self.save_name(m.group(1).capitalize())
            return f"Приятно познакомиться, {self.name}! Запомнил."

        calc = calculate(text)
        if calc:
            self.last_tag = "калькулятор"
            return calc

        v = self.vectorize(text)
        if v.sum() == 0:
            return "Не понял тебя. Напиши «помощь», чтобы узнать, что я умею."
        _, p = self.forward(v[None, :])
        p = p[0]
        if p.max() < 0.5:
            return "Не уверен, что понял. Попробуй сказать иначе."
        tag = self.tags[int(p.argmax())]
        self.last_tag = tag
        if tag == "время":
            return datetime.now().strftime("Сейчас %H:%M")
        if tag == "моё_имя":
            if self.name:
                return f"Тебя зовут {self.name}."
            return "Пока не знаю. Скажи: «меня зовут ...»"
        answer = random.choice(self.intents[tag]["responses"])
        if tag in ("приветствие", "прощание") and self.name:
            answer = answer.rstrip("!.?") + f", {self.name}!"
        return answer


if __name__ == "__main__":
    bot = ChatBot(INTENTS)
    print("Бот запущен. Для выхода напиши «пока».")
    while True:
        try:
            text = input("Ты: ")
        except EOFError:
            break
        print("Бот:", bot.reply(text))
        if bot.last_tag == "прощание":
            break
