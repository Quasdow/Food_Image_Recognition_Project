import tensorflow as tf

class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros',
    'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee',
    'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings',
    'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon',
    'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup',
    'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt',
    'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese',
    'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings',
    'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich',
    'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi',
    'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

def get_classes() -> list[str]:
    """
    Return the list of Food101 class names.
    """
    return class_names

def load_and_prep(image_bytes: bytes, shape: int = 224, scale: bool = True) -> tf.Tensor:
    """
    Decode raw image bytes, resize to (shape, shape),
    and normalize to [0,1] if scale=True.
    """
    try:
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        tf.ensure_shape(image, [None, None, 3])
        image = tf.image.resize(image, [shape, shape])
        if scale:
            image = image / 255.0
        return image
    except Exception as e:
        raise ValueError(f"Error decoding or processing image: {e}")