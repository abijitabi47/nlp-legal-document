
import re

CATEGORY_KEYWORDS = {
    "Domestic Violence": [
        "domestic violence",
        "protection order",
        "protection of women from domestic violence act",
        "shared household",
        "domestic incident report",
        "economic abuse",
        "residence order",
        "section 12 of the domestic violence act",
        "section 17 of the domestic violence act",
    ],
    "Criminal Law": [
        "criminal case",
        "criminal appeal",
        "criminal revision",
        "accused",
        "prosecution",
        "conviction",
        "acquittal",
        "criminal procedure",
        "code of criminal procedure",
        "bail application",
    ],
    "Family and Matrimonial Law": [
        "divorce",
        "matrimonial dispute",
        "maintenance",
        "child custody",
        "marital cruelty",
        "restitution of conjugal rights",
        "family court",
        "dowry",
    ],
    "Property and Land Disputes": [
        "property dispute",
        "land ownership",
        "title deed",
        "land possession",
        "property registration",
        "partition suit",
        "sale deed",
        "eviction",
        "land acquisition",
    ],
    "Cybercrime": [
        "cybercrime",
        "cyber crime",
        "online fraud",
        "identity theft",
        "phishing",
        "unauthorised access",
        "information technology act",
        "electronic evidence",
        "data breach",
    ],
    "Consumer Law": [
        "consumer protection act",
        "deficiency in service",
        "defective goods",
        "consumer dispute",
        "unfair trade practice",
        "consumer commission",
    ],
    "Constitutional Law": [
        "fundamental rights",
        "article 14",
        "article 19",
        "article 21",
        "constitutional validity",
        "writ petition",
        "article 226",
        "article 32",
    ],
    "Labour and Employment Law": [
        "wrongful termination",
        "industrial dispute",
        "labour court",
        "back wages",
        "employment dispute",
        "workman",
        "retrenchment",
        "workplace harassment",
    ],
    "Contract and Commercial Law": [
        "breach of contract",
        "contractual obligation",
        "specific performance",
        "commercial dispute",
        "arbitration agreement",
        "damages for breach",
        "contract act",
    ],
    "Sexual Offences and Crimes Against Women": [
        "sexual assault",
        "sexual violence",
        "rape",
        "molestation",
        "sexual harassment",
        "offence against women",
        "pocso act",
    ],
    "Corruption and Financial Crime": [
        "money laundering",
        "prevention of corruption act",
        "corruption",
        "bribery",
        "fraudulent transactions",
        "criminal breach of trust",
        "benami property",
    ],
}


def detect_categories(text, min_matches=1):
    """
    Return matching legal categories and their keyword matches.

    This is a rule-based baseline, not a trained classifier.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    detected = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        matches = [
            keyword
            for keyword in keywords
            if re.search(
                r"(?<!\w)" + re.escape(keyword) + r"(?!\w)",
                text
            )
        ]

        if len(matches) >= min_matches:
            detected[category] = matches

    return detected


if __name__ == "__main__":
    sample = """
    The applicant sought a protection order under the
    Protection of Women from Domestic Violence Act.
    The court also considered maintenance.
    """

    results = detect_categories(sample)

    if results:
        for category, matches in results.items():
            print(f"{category}: {', '.join(matches)}")
    else:
        print("No category detected.")
