import random
import sys

_INTRO_TEMPLATES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
]

_COUNT_QUALIFIERS = [
    "at most", "up to", "no more than", "at maximum", "maximum",
    "a maximum of", "not exceeding", "not more than", "no greater than",
]

_DIGIT_DESCRIPTORS = [
    "up to {d} digits each",
    "maximum {d} digits each",
    "{d} digits or fewer each",
    "{d} digits max each",
    "no more than {d} digits each",
    "at most {d} digits each",
    "with {d} digits or less each",
    "{d}-digit numbers or smaller",
]

_INSTRUCTION_TEMPLATES = [
    "Add {q} {n} new numbers ({dd}) that continue the sequence.",
    "Please add {q} {n} more numbers ({dd}) to continue it.",
    "Add {q} {n} more values ({dd}) to continue the sequence.",
    "Create {q} {n} more numbers ({dd}) that follow.",
    "Write {q} {n} more numbers ({dd}) in this sequence.",
    "Generate {q} {n} additional numbers ({dd}).",
    "Extend it with {q} {n} new numbers ({dd}).",
    "Continue the sequence with {q} {n} more numbers ({dd})",
    "Create {q} {n} more numbers ({dd}) to extend the sequence",
    "Generate {q} {n} additional numbers ({dd}) to extend this sequence",
]

_FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
]

_NO_EXPLAIN_SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
]


def generate_diverse_prompt(rng: random.Random, *, min_examples: int = 3,
                            max_examples: int = 9, min_value: int = 100,
                            max_value: int = 999, answer_count: int = 10,
                            max_digits: int = 3) -> str:
    n_examples = rng.randint(min_examples, max_examples)
    examples = ", ".join(str(rng.randint(min_value, max_value)) for _ in range(n_examples))
    intro = rng.choice(_INTRO_TEMPLATES).format(examples=examples)
    qualifier = rng.choice(_COUNT_QUALIFIERS)
    digit_desc = rng.choice(_DIGIT_DESCRIPTORS).format(d=max_digits)
    instruction = rng.choice(_INSTRUCTION_TEMPLATES).format(q=qualifier, n=answer_count, dd=digit_desc)
    fmt = rng.choice(_FORMAT_SUFFIXES)
    suffix = rng.choice(_NO_EXPLAIN_SUFFIXES)
    return f"{intro} {instruction} {fmt} {suffix}"

# python prompts/numbers.py 10000 > prompts/user-numbers-10k.txt
if __name__ == "__main__":
    n = int(sys.argv[1])
    rng = random.Random()
    for _ in range(n):
        print(generate_diverse_prompt(rng))
