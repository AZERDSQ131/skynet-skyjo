"""Distribution des cartes Skyjo et utilitaires liés aux valeurs."""

CARD_COUNTS = {
    -2: 5,
    -1: 10,
    0: 15,
}
for v in range(1, 13):
    CARD_COUNTS[v] = 10

MIN_VALUE = -2
MAX_VALUE = 12
NUM_VALUES = MAX_VALUE - MIN_VALUE + 1  # 15

TOTAL_CARDS = sum(CARD_COUNTS.values())  # 150


def build_deck():
    deck = []
    for value, count in CARD_COUNTS.items():
        deck.extend([value] * count)
    return deck


def value_to_index(value):
    return value - MIN_VALUE


def index_to_value(index):
    return index + MIN_VALUE
