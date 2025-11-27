import numpy as np


def get_color_name(average_color):
    """
    Determine color name from BGR values (OpenCV format).

    Args:
        average_color: numpy array with BGR values [B, G, R]

    Returns:
        String with color name in Polish or English
    """
    # Kolory w formacie BGR (jak w OpenCV)
    color_names = {
        # Podstawowe kolory
        (255, 0, 0): "Niebieski",      # Blue
        (0, 255, 0): "Zielony",        # Green
        (0, 0, 255): "Czerwony",       # Red
        (255, 255, 255): "Biały",      # White
        (0, 0, 0): "Czarny",           # Black

        # Dodatkowe kolory
        (0, 255, 255): "Żółty",        # Yellow
        (0, 165, 255): "Pomarańczowy", # Orange
        (128, 0, 128): "Fioletowy",    # Purple
        (203, 192, 255): "Różowy",     # Pink
        (128, 128, 128): "Szary",      # Gray

        # Odcienie niebieskiego
        (255, 255, 0): "Cyan",         # Cyan
        (128, 0, 0): "Ciemnoniebieski", # Dark Blue
        (255, 144, 30): "Błękitny",    # Dodger Blue

        # Odcienie zielonego
        (0, 128, 0): "Ciemnozielony",  # Dark Green
        (47, 255, 173): "Jasnozielony", # Green Yellow

        # Odcienie czerwonego
        (0, 0, 128): "Bordowy",        # Maroon
        (0, 0, 139): "Ciemnoczerwony", # Dark Red

        # Odcienie brązowego
        (19, 69, 139): "Brązowy",      # Saddle Brown
        (42, 42, 165): "Jasnobrązowy", # Brown

        # Inne
        (211, 0, 148): "Fuksja",       # Magenta/Fuchsia
        (130, 0, 75): "Indygo",        # Indigo
        (32, 165, 218): "Złoty",       # Gold
        (0, 215, 255): "Złoty",        # Gold alternative
        (224, 255, 255): "Beżowy",     # Beige
    }

    min_distance = float('inf')
    closest_color = "Nieznany"

    # Konwertuj average_color do numpy array jeśli jeszcze nie jest
    if not isinstance(average_color, np.ndarray):
        average_color = np.array(average_color)

    for color_bgr, name in color_names.items():
        # Oblicz odległość euklidesową w przestrzeni BGR
        distance = np.linalg.norm(np.array(color_bgr) - average_color)
        if distance < min_distance:
            min_distance = distance
            closest_color = name

    return closest_color


def get_color_name_advanced(average_color):
    """
    Bardziej zaawansowana wersja z progami dla lepszego rozpoznawania.
    """
    b, g, r = average_color[0], average_color[1], average_color[2]

    # Jasność (brightness)
    brightness = (int(r) + int(g) + int(b)) / 3

    # Bardzo ciemne kolory
    if brightness < 40:
        return "Czarny"

    # Bardzo jasne kolory
    if brightness > 220 and abs(r - g) < 20 and abs(g - b) < 20:
        return "Biały"

    # Odcienie szarości
    if abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
        if brightness < 85:
            return "Ciemnoszary"
        elif brightness < 170:
            return "Szary"
        else:
            return "Jasnoszary"

    # Teraz sprawdź dominujący kanał
    max_channel = max(r, g, b)
    min_channel = min(r, g, b)

    # Czerwony dominuje
    if r == max_channel and r > g + 30 and r > b + 30:
        if g > b + 20:
            return "Pomarańczowy"
        elif b > g + 20:
            return "Różowy"
        else:
            return "Czerwony"

    # Zielony dominuje
    elif g == max_channel and g > r + 30 and g > b + 30:
        if r > b + 20:
            return "Żółtozielony"
        else:
            return "Zielony"

    # Niebieski dominuje
    elif b == max_channel and b > r + 30 and b > g + 30:
        if r > g + 20:
            return "Fioletowy"
        elif g > r + 20:
            return "Cyan"
        else:
            return "Niebieski"

    # Żółty (czerwony + zielony)
    elif r > 150 and g > 150 and b < 100:
        return "Żółty"

    # Cyan (zielony + niebieski)
    elif g > 150 and b > 150 and r < 100:
        return "Cyan"

    # Magenta/Fioletowy (czerwony + niebieski)
    elif r > 150 and b > 150 and g < 100:
        return "Fioletowy"

    # Brązowy
    elif r > g and g > b and r < 180:
        return "Brązowy"

    return "Mieszany"