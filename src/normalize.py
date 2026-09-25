import re
import unicodedata


def normalize_text(text):
    """
    Basic Unicode-safe normalization.
    Preserves characters from Indian and other scripts.
    """

    if text is None:
        return ""

    text = str(text).strip().lower()

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Replace & with "and"
    text = text.replace("&", " and ")

    # Replace punctuation/symbols while preserving Unicode letters
    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        # Keep letters, numbers and combining marks
        if category.startswith(("L", "N", "M")):
            cleaned.append(char)

        # Convert punctuation/symbols to spaces
        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(name):
    return normalize_text(name)


def normalize_address(address):
    return normalize_text(address)


if __name__ == "__main__":

    examples = [
        "Payne Énterprises, LLC",
        "3315 FREMONT ST, PEORIA, IL",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
        "Raj Investments LLP",
        "PAYNE-ENRTPRMISES",
    ]

    for x in examples:
        print("Original  :", x)
        print("Normalized:", normalize_text(x))
        print()