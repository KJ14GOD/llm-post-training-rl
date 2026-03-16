import random

PROMPT_QUESTIONS = [
    # Level 1: Basic Arithmetic (67)
    "What is 12 + 19? Answer with only the integer.",
    "What is 45 - 18? Answer with only the integer.",
    "What is 23 * 17? Answer with only the integer.",
    "What is 144 / 12? Answer with only the integer.",
    "What is 56 + 37? Answer with only the integer.",
    "What is 92 - 44? Answer with only the integer.",
    "What is 34 * 12? Answer with only the integer.",
    "What is 210 / 14? Answer with only the integer.",
    "What is 63 + 28? Answer with only the integer.",
    "What is 81 - 36? Answer with only the integer.",
    "What is 17 * 18? Answer with only the integer.",
    "What is 196 / 7? Answer with only the integer.",
    "What is 44 + 39? Answer with only the integer.",
    "What is 88 - 41? Answer with only the integer.",
    "What is 25 * 16? Answer with only the integer.",
    "What is 240 / 15? Answer with only the integer.",
    "What is 73 + 24? Answer with only the integer.",
    "What is 65 - 27? Answer with only the integer.",
    "What is 19 * 14? Answer with only the integer.",
    "What is 156 / 12? Answer with only the integer.",
    "What is 48 + 57? Answer with only the integer.",
    "What is 91 - 53? Answer with only the integer.",
    "What is 21 * 13? Answer with only the integer.",
    "What is 132 / 11? Answer with only the integer.",
    "What is 67 + 38? Answer with only the integer.",
    "What is 84 - 26? Answer with only the integer.",
    "What is 27 * 11? Answer with only the integer.",
    "What is 168 / 12? Answer with only the integer.",
    "What is 36 + 29? Answer with only the integer.",
    "What is 75 - 48? Answer with only the integer.",
    "What is 31 * 12? Answer with only the integer.",
    "What is 180 / 15? Answer with only the integer.",
    "What is 49 + 63? Answer with only the integer.",
    "What is 93 - 37? Answer with only the integer.",
    "What is 22 * 19? Answer with only the integer.",
    "What is 154 / 14? Answer with only the integer.",
    "What is 57 + 41? Answer with only the integer.",
    "What is 88 - 52? Answer with only the integer.",
    "What is 24 * 15? Answer with only the integer.",
    "What is 198 / 9? Answer with only the integer.",
    "What is 62 + 47? Answer with only the integer.",
    "What is 99 - 44? Answer with only the integer.",
    "What is 18 * 17? Answer with only the integer.",
    "What is 176 / 16? Answer with only the integer.",
    "What is 54 + 36? Answer with only the integer.",
    "What is 83 - 29? Answer with only the integer.",
    "What is 16 * 21? Answer with only the integer.",
    "What is 220 / 20? Answer with only the integer.",
    "What is 68 + 34? Answer with only the integer.",
    "What is 74 - 39? Answer with only the integer.",
    "What is 26 * 14? Answer with only the integer.",
    "What is 162 / 9? Answer with only the integer.",
    "What is 59 + 28? Answer with only the integer.",
    "What is 81 - 47? Answer with only the integer.",
    "What is 28 * 13? Answer with only the integer.",
    "What is 144 / 18? Answer with only the integer.",
    "What is 71 + 25? Answer with only the integer.",
    "What is 64 - 33? Answer with only the integer.",
    "What is 15 * 22? Answer with only the integer.",
    "What is 200 / 25? Answer with only the integer.",
    "What is 43 + 52? Answer with only the integer.",
    "What is 97 - 58? Answer with only the integer.",
    "What is 33 * 12? Answer with only the integer.",
    "What is 216 / 12? Answer with only the integer.",
    "What is 58 + 46? Answer with only the integer.",
    "What is 90 - 41? Answer with only the integer.",
    "What is 27 * 16? Answer with only the integer.",

    # Level 2: Multi-Step Arithmetic (67)
    "What is (12 + 19) * 3? Answer with only the integer.",
    "What is (45 - 18) * 5? Answer with only the integer.",
    "What is (23 * 17) + 120? Answer with only the integer.",
    "What is (144 / 12) * 7? Answer with only the integer.",
    "What is (56 + 37) - 29? Answer with only the integer.",
    "What is (92 - 44) * 4? Answer with only the integer.",
    "What is (34 * 12) + 88? Answer with only the integer.",
    "What is (210 / 14) * 6? Answer with only the integer.",
    "What is (63 + 28) * 2? Answer with only the integer.",
    "What is (81 - 36) * 3? Answer with only the integer.",
    "What is (17 * 18) + 50? Answer with only the integer.",
    "What is (196 / 7) * 5? Answer with only the integer.",
    "What is (44 + 39) * 4? Answer with only the integer.",
    "What is (88 - 41) * 6? Answer with only the integer.",
    "What is (25 * 16) + 200? Answer with only the integer.",
    "What is (240 / 15) * 9? Answer with only the integer.",
    "What is (73 + 24) * 3? Answer with only the integer.",
    "What is (65 - 27) * 7? Answer with only the integer.",
    "What is (19 * 14) + 30? Answer with only the integer.",
    "What is (156 / 12) * 8? Answer with only the integer.",
    "What is (48 + 57) * 2? Answer with only the integer.",
    "What is (91 - 53) * 6? Answer with only the integer.",
    "What is (21 * 13) + 100? Answer with only the integer.",
    "What is (132 / 11) * 9? Answer with only the integer.",
    "What is (67 + 38) * 4? Answer with only the integer.",
    "What is (84 - 26) * 3? Answer with only the integer.",
    "What is (27 * 11) + 50? Answer with only the integer.",
    "What is (168 / 12) * 10? Answer with only the integer.",
    "What is (36 + 29) * 5? Answer with only the integer.",
    "What is (75 - 48) * 9? Answer with only the integer.",
    "What is (31 * 12) + 70? Answer with only the integer.",
    "What is (180 / 15) * 8? Answer with only the integer.",
    "What is (49 + 63) * 2? Answer with only the integer.",
    "What is (93 - 37) * 4? Answer with only the integer.",
    "What is (22 * 19) + 200? Answer with only the integer.",
    "What is (154 / 14) * 7? Answer with only the integer.",
    "What is (57 + 41) * 3? Answer with only the integer.",
    "What is (88 - 52) * 5? Answer with only the integer.",
    "What is (24 * 15) + 60? Answer with only the integer.",
    "What is (198 / 9) * 4? Answer with only the integer.",
    "What is (62 + 47) * 2? Answer with only the integer.",
    "What is (99 - 44) * 6? Answer with only the integer.",
    "What is (18 * 17) + 90? Answer with only the integer.",
    "What is (176 / 16) * 9? Answer with only the integer.",
    "What is (54 + 36) * 3? Answer with only the integer.",
    "What is (83 - 29) * 5? Answer with only the integer.",
    "What is (16 * 21) + 80? Answer with only the integer.",
    "What is (220 / 20) * 6? Answer with only the integer.",
    "What is (68 + 34) * 4? Answer with only the integer.",
    "What is (74 - 39) * 7? Answer with only the integer.",
    "What is (26 * 14) + 50? Answer with only the integer.",
    "What is (162 / 9) * 5? Answer with only the integer.",
    "What is (59 + 28) * 6? Answer with only the integer.",
    "What is (81 - 47) * 8? Answer with only the integer.",
    "What is (28 * 13) + 60? Answer with only the integer.",
    "What is (144 / 18) * 9? Answer with only the integer.",
    "What is (71 + 25) * 4? Answer with only the integer.",
    "What is (64 - 33) * 6? Answer with only the integer.",
    "What is (15 * 22) + 100? Answer with only the integer.",
    "What is (200 / 25) * 12? Answer with only the integer.",
    "What is (43 + 52) * 3? Answer with only the integer.",
    "What is (97 - 58) * 8? Answer with only the integer.",
    "What is (33 * 12) + 40? Answer with only the integer.",
    "What is (216 / 12) * 7? Answer with only the integer.",
    "What is (58 + 46) * 2? Answer with only the integer.",
    "What is (90 - 41) * 5? Answer with only the integer.",
    "What is (27 * 16) + 100? Answer with only the integer.",

    # Level 3: Word Problems (66)
    "A train travels 60 miles per hour for 3 hours. How far does it travel? Answer with only the integer.",
    "A box contains 24 packs of pencils and each pack has 8 pencils. How many pencils are there total? Answer with only the integer.",
    "Sarah buys 5 notebooks for 7 dollars each and 3 pens for 2 dollars each. What is the total cost? Answer with only the integer.",
    "A warehouse stores 48 boxes with 36 items in each box. How many items are stored? Answer with only the integer.",
    "A farmer has 125 apples and sells 47 of them. How many apples remain? Answer with only the integer.",
    "A classroom has 28 students and each student receives 3 books. How many books are distributed? Answer with only the integer.",
    "A car drives 72 miles per hour for 4 hours. How far does it travel? Answer with only the integer.",
    "A store sells 18 boxes of cookies with 24 cookies per box. How many cookies were sold? Answer with only the integer.",
    "A school buys 36 desks and each desk costs 45 dollars. What is the total cost? Answer with only the integer.",
    "A delivery truck carries 125 packages and delivers 38. How many packages remain? Answer with only the integer.",
    "A cyclist rides 15 miles per hour for 4 hours. How far does the cyclist travel? Answer with only the integer.",
    "A baker makes 12 trays of cookies with 20 cookies on each tray. How many cookies are made? Answer with only the integer.",
    "A theater sells 35 tickets for each of 8 shows. How many tickets are sold? Answer with only the integer.",
    "A farmer plants 25 rows of corn with 18 plants in each row. How many plants are planted? Answer with only the integer.",
    "A delivery company delivers 45 packages per hour for 6 hours. How many packages are delivered? Answer with only the integer.",
    "A library has 120 books and buys 35 more. How many books does it have now? Answer with only the integer.",
    "A runner runs 8 miles per day for 7 days. How many miles total? Answer with only the integer.",
    "A factory produces 75 toys per day for 4 days. How many toys are produced? Answer with only the integer.",
    "A bus holds 50 passengers and makes 6 trips. How many passengers total? Answer with only the integer.",
    "A gardener plants 9 rows with 14 flowers each. How many flowers are planted? Answer with only the integer.",
    "A student reads 18 pages per day for 12 days. How many pages are read? Answer with only the integer.",
    "A warehouse ships 64 boxes each day for 5 days. How many boxes are shipped? Answer with only the integer.",
    "A bakery sells 36 loaves of bread each morning for 7 mornings. How many loaves are sold? Answer with only the integer.",
    "A farmer collects 28 eggs each day for 9 days. How many eggs are collected? Answer with only the integer.",
    "A painter paints 12 rooms each week for 4 weeks. How many rooms are painted? Answer with only the integer.",
    "A train carries 120 passengers and drops off 35. How many remain on the train? Answer with only the integer.",
    "A bookstore sells 45 books in the morning and 38 in the afternoon. How many books are sold total? Answer with only the integer.",
    "A chef cooks 15 meals each hour for 6 hours. How many meals are cooked? Answer with only the integer.",
    "A farmer has 84 cows and sells 19. How many cows remain? Answer with only the integer.",
    "A teacher gives 4 pencils to each of 23 students. How many pencils are given out? Answer with only the integer.",
    "A company prints 250 flyers and distributes 175. How many flyers remain? Answer with only the integer.",
    "A swimmer swims 40 laps each day for 6 days. How many laps are swum? Answer with only the integer.",
    "A farmer grows 16 rows of tomatoes with 20 plants in each row. How many plants are there? Answer with only the integer.",
    "A truck carries 18 crates with 12 bottles each. How many bottles are there? Answer with only the integer.",
    "A baker makes 24 cakes and sells 17. How many cakes remain? Answer with only the integer.",
    "A store receives 90 shirts and sells 34. How many shirts remain? Answer with only the integer.",
    "A runner completes 12 races each year for 5 years. How many races total? Answer with only the integer.",
    "A driver travels 55 miles per hour for 6 hours. How far does the driver travel? Answer with only the integer.",
    "A farmer has 45 sheep and buys 18 more. How many sheep now? Answer with only the integer.",
    "A student solves 15 problems per day for 10 days. How many problems are solved? Answer with only the integer.",
    "A warehouse stacks 12 boxes in each row with 15 rows. How many boxes total? Answer with only the integer.",
    "A baker bakes 32 muffins each morning for 5 mornings. How many muffins are baked? Answer with only the integer.",
    "A delivery driver drops off 18 packages per hour for 8 hours. How many packages are delivered? Answer with only the integer.",
    "A factory makes 90 parts each day for 3 days. How many parts are made? Answer with only the integer.",
    "A student buys 6 pens for 3 dollars each. What is the total cost? Answer with only the integer.",
    "A gardener waters 24 plants each day for 7 days. How many plant waterings occur? Answer with only the integer.",
    "A store sells 12 bikes each month for 9 months. How many bikes are sold? Answer with only the integer.",
    "A library lends 40 books each day for 6 days. How many books are lent? Answer with only the integer.",
    "A teacher grades 18 tests per hour for 4 hours. How many tests are graded? Answer with only the integer.",
    "A worker packs 36 boxes per shift for 5 shifts. How many boxes are packed? Answer with only the integer.",
    "A bus drives 50 miles each trip and makes 7 trips. How many miles total? Answer with only the integer.",
    "A farmer harvests 28 baskets with 15 apples each. How many apples total? Answer with only the integer.",
    "A chef prepares 14 meals each hour for 5 hours. How many meals are prepared? Answer with only the integer.",
    "A bookstore receives 120 books and sells 45. How many books remain? Answer with only the integer.",
    "A delivery company sends out 16 trucks each carrying 20 packages. How many packages are sent? Answer with only the integer.",
    "A school has 30 classrooms with 25 students each. How many students total? Answer with only the integer.",
    "A farmer plants 14 rows of potatoes with 18 plants in each row. How many plants are planted? Answer with only the integer.",
    "A store sells 25 shirts each day for 8 days. How many shirts are sold? Answer with only the integer.",
    "A factory produces 32 machines each week for 6 weeks. How many machines are produced? Answer with only the integer.",
    "A baker sells 18 cakes each day for 9 days. How many cakes are sold? Answer with only the integer.",
    "A delivery driver makes 12 stops each hour for 5 hours. How many stops are made? Answer with only the integer.",
    "A gardener plants 16 rows with 12 flowers each. How many flowers are planted? Answer with only the integer.",
    "A farmer collects 32 baskets of oranges with 18 oranges each. How many oranges total? Answer with only the integer.",
    "A bus makes 9 trips carrying 40 passengers each trip. How many passengers total? Answer with only the integer.",
    "A worker builds 14 tables each week for 6 weeks. How many tables are built? Answer with only the integer.",
    "A store sells 48 notebooks each day for 5 days. How many notebooks are sold? Answer with only the integer.",
    "A farmer harvests 18 baskets of apples with 24 apples in each basket. How many apples are harvested in total? Answer with only the integer."
]

GROUND_TRUTH = [
   # Level 1 answers (67)
    31, 27, 391, 12, 93, 48, 408, 15, 91, 45,
    306, 28, 83, 47, 400, 16, 97, 38, 266, 13,
    105, 38, 273, 12, 105, 58, 297, 14, 65, 27,
    372, 12, 112, 56, 418, 11, 98, 36, 360, 22,
    109, 55, 306, 11, 90, 54, 336, 11, 102, 35,
    364, 18, 87, 34, 364, 8, 96, 31, 330, 8,
    95, 39, 396, 18, 104, 49, 432,

    # Level 2 answers (67)
    93, 135, 511, 84, 64, 192, 496, 90, 182, 135,
    356, 140, 332, 282, 600, 144, 291, 266, 296, 104,
    210, 228, 373, 108, 420, 174, 347, 140, 325, 243,
    442, 96, 224, 224, 618, 77, 294, 180, 420, 88,
    218, 330, 396, 99, 270, 270, 416, 66, 408, 245,
    414, 90, 522, 272, 424, 72, 384, 186, 430, 96,
    285, 312, 436, 126, 208, 245, 532,

    # Level 3 answers (67)
    180, 192, 41, 1728, 78, 84, 288, 432, 1620, 87,
    60, 240, 280, 450, 270, 155, 56, 300, 300, 126,
    216, 320, 252, 252, 48, 85, 83, 90, 65, 92,
    75, 240, 320, 216, 7, 56, 60, 330, 63, 150,
    180, 160, 144, 270, 18, 168, 108, 240, 72, 180,
    350, 420, 70, 75, 320, 750, 252, 200, 192, 162,
    60, 192, 576, 360, 84, 240, 432
]

LEVEL_NAMES = ["Level 1: Basic Arithmetic", "Level 2: Multi-Step", "Level 3: Word Problems"]

SPLIT_SEED = 42
LEVEL_SIZE = 67
TRAIN_PER_LEVEL = 53
EVAL_PER_LEVEL = LEVEL_SIZE - TRAIN_PER_LEVEL


def _build_stratified_split():
    assert len(PROMPT_QUESTIONS) == len(GROUND_TRUTH) == LEVEL_SIZE * len(LEVEL_NAMES)

    train_questions = []
    train_answers = []
    train_level_ids = []
    eval_questions = []
    eval_answers = []
    eval_level_ids = []

    for level_idx in range(len(LEVEL_NAMES)):
        start = level_idx * LEVEL_SIZE
        end = start + LEVEL_SIZE
        level_examples = list(zip(PROMPT_QUESTIONS[start:end], GROUND_TRUTH[start:end]))

        rng = random.Random(SPLIT_SEED + level_idx)
        rng.shuffle(level_examples)

        train_examples = level_examples[:TRAIN_PER_LEVEL]
        eval_examples = level_examples[TRAIN_PER_LEVEL:]

        train_questions.extend(question for question, _ in train_examples)
        train_answers.extend(answer for _, answer in train_examples)
        train_level_ids.extend([level_idx] * len(train_examples))

        eval_questions.extend(question for question, _ in eval_examples)
        eval_answers.extend(answer for _, answer in eval_examples)
        eval_level_ids.extend([level_idx] * len(eval_examples))

    return (
        train_questions,
        train_answers,
        train_level_ids,
        eval_questions,
        eval_answers,
        eval_level_ids,
    )


(
    TRAIN_PROMPT_QUESTIONS,
    TRAIN_GROUND_TRUTH,
    TRAIN_LEVEL_IDS,
    EVAL_PROMPT_QUESTIONS,
    EVAL_GROUND_TRUTH,
    EVAL_LEVEL_IDS,
) = _build_stratified_split()

TRAIN_LEVEL_COUNTS = [TRAIN_PER_LEVEL] * len(LEVEL_NAMES)
EVAL_LEVEL_COUNTS = [EVAL_PER_LEVEL] * len(LEVEL_NAMES)
