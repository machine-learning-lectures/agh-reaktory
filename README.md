# AGH Reaktory: optymalizacja w Pythonie

Pięć mikro-labów dla studentów AGH pokazujących metody optymalizacji na
syntetycznym case'ie postoju paliwowo-remontowego bloku jądrowego.

## Tematy

1. Greedy jako szybki baseline.
2. LP-relaxation i MILP w Pyomo.
3. Local search i 2-opt dla tras ekip.
4. Simulated annealing w czystym Pythonie.
5. Pareto optimization dla czasu, kosztu i dawki.

## Uruchomienie

```bash
uv run pytest
uv run quarto render
```

Quarto renderuje stronę do katalogu `docs/`, który można wskazać jako źródło
GitHub Pages.

## Struktura

- `src/utilis/library.py` - wspólne funkcje dydaktyczne i modele Pyomo.
- `labs/` - źródła mikro-labów Quarto.
- `tests/` - testy jednostkowe dla algorytmów i małych modeli.
- `doc/` - notatki źródłowe i materiały kontekstowe.
