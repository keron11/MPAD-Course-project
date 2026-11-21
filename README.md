# Аналіз продажів і комерційної діяльності підприємства

Курсова робота з дисципліни **«Мови програмування для аналізу даних»**  
Спеціальність 122 «Комп’ютерні науки» — Національний університет «Полтавська політехніка імені Юрія Кондратюка».

## Мета проєкту
Розробити відтворюваний програмний проєкт на базі Python для завантаження, очищення, збагачення, статистичного аналізу та візуалізації даних про продажі роздрібного підприємства (Sample Superstore), а також побудувати інтерактивну аналітичну панель у Tableau Public.

## Джерела даних
- `data/raw/SampleSuperstore.csv` — відкритий набір даних **Sample Superstore** (Kaggle / Tableau Public Sample Data), 9 994 записи, 13 ознак (категорії, підкатегорії, географія, сегменти, способи доставки, продажі, кількість, знижка, прибуток).
- `data/raw/regional_managers.csv` — довідник регіональних менеджерів і логістичних хабів (відповідно до стандартної таблиці `People` набору Superstore).
- `data/raw/shipping_modes.csv` — довідник параметрів способів доставки (SLA у днях, пріоритет, базова вартість).

## Структура репозиторію
```text
course-project/
├── data/
│   ├── raw/                  # Вихідні дані та довідники
│   └── processed/            # Очищений та збагачений набір даних (superstore_cleaned.csv)
├── notebooks/
│   └── analysis.ipynb        # Основний Jupyter Notebook із повним циклом аналізу
├── src/
│   └── run_analysis.py       # Скрипт повного відтворення розрахунків і графіків
├── results/
│   ├── figures/              # 7 графіків Matplotlib / Seaborn (300 dpi)
│   └── exports/              # Агреговані та зведені таблиці в CSV / JSON
├── README.md
├── requirements.txt
└── .gitignore
```

## Використані технології
- Python 3.10+
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook
- Tableau Public

## Інструкція із запуску
1. Встановіть залежності:
   ```bash
   pip install -r requirements.txt
   ```
2. Запустіть повний цикл аналізу та генерації графіків:
   ```bash
   python src/run_analysis.py
   ```
3. Або відкрийте та послідовно виконайте ноутбук:
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```
4. Очищений файл `data/processed/superstore_cleaned.csv` готовий для імпорту в Tableau Public.
