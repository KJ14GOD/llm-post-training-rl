PROMPT_QUESTIONS = [
    # Level 1: Basic Arithmetic
    "What is 17 + 28?",
    "What is 42 - 19?",
    "What is 36 * 14?",
    "What is 144 / 12?",
    "What is 29 + 57?",
    "What is 81 - 46?",
    "What is 23 * 17?",
    "What is 196 / 14?",
    "What is 45 + 38?",
    "What is 72 - 39?",
    # Level 2: Multi-Step Arithmetic
    "What is (17 + 28) * 4?",
    "What is (42 - 19) * 6?",
    "What is (36 * 14) + 120?",
    "What is (144 / 12) * 9?",
    "What is (29 + 57) - 33?",
    "What is (81 - 46) * 7?",
    "What is (23 * 17) + (45 + 38)?",
    "What is (196 / 14) * 5 + 20?",
    "What is (45 + 38) * 3 - 50?",
    "What is (72 - 39) * 8 + 16?",
    # Level 3: Word Problems
    "A train travels 60 miles per hour for 3 hours. How far does it travel?",
    "A box contains 24 packs of pencils and each pack has 8 pencils. How many pencils are there in total?",
    "Sarah buys 5 notebooks for 7 dollars each and 3 pens for 2 dollars each. What is the total cost in dollars?",
    "A warehouse stores 48 boxes and each box contains 36 items. How many items are stored in total?",
    "A farmer has 125 apples and sells 47 of them. How many apples remain?",
    "A classroom has 28 students and each student receives 3 books. How many books are distributed in total?",
    "A car travels 72 miles per hour for 4 hours. How far does it travel?",
    "A store sells 18 boxes of cookies and each box contains 24 cookies. How many cookies were sold?",
    "A school buys 36 desks and each desk costs 45 dollars. What is the total cost in dollars?",
    "A delivery truck carries 125 packages and delivers 38 of them. How many packages remain?"
]

GROUND_TRUTH = [
    45, 23, 504, 12, 86, 35, 391, 14, 83, 33,
    180, 138, 624, 108, 53, 245, 474, 90, 199, 280,
    180, 192, 41, 1728, 78, 84, 288, 432, 1620, 87,
]

LEVEL_NAMES = ["Level 1: Basic Arithmetic", "Level 2: Multi-Step", "Level 3: Word Problems"]
