# Гид по решению

### Ноутбук
- В оглавлении описано само решение и ход работы
- Суть решения - классический QA, поиск токенов ответа в контексте
- Далее в ноубуке по пунктам разбирается, что было сделано и как
- В выводах я предположил, что позволило мне добиться чудесного f1-score в 8.45%)))

### Телеграмм бот
1. Бот запускается из файла main.py
2. Бот запускает модель при выполнении команды /start
3. Боту можно подать контекст, в котором он будет исктаь ответы
4. Любое сообщение боту - вопрос к контексту, если оно не является командой
