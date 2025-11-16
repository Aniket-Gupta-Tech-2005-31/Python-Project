import itertools
import random

deck_of_cards = list(itertools.product(range(1, 14), ['Spade', 'Heart', 'Diamond', 'Club']))

random.shuffle(deck_of_cards)

print("You got:")
for i in range(5):
    print(deck_of_cards[i][0], "of", deck_of_cards[i][1])
