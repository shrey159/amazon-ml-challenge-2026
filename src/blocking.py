from normalize import normalize_name, normalize_address


# --------------------------------------------------
# Generic business-name words
# --------------------------------------------------

COMMON_NAME_TOKENS = {
    "the",
    "and",
    "company",
    "co",
    "corporation",
    "corp",
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "pvt",
    "private",
    "llp",
    "plc",
    "group",
    "india",
}


# --------------------------------------------------
# Generic address words
# --------------------------------------------------

COMMON_ADDRESS_TOKENS = {
    "road",
    "street",
    "st",
    "rd",
    "avenue",
    "ave",
    "lane",
    "ln",
    "drive",
    "dr",
    "highway",
    "hwy",
    "boulevard",
    "blvd",
    "building",
    "floor",
    "block",
    "near",
    "beside",
    "opposite",
    "district",
    "city",
    "town",
    "village",
    "state",
    "india",
    "usa",
    "us",
}


# --------------------------------------------------
# Name tokens
# --------------------------------------------------

def get_name_tokens(name):
    """
    Get informative tokens from a business name.
    """

    normalized = normalize_name(name)

    tokens = normalized.split()

    useful_tokens = []

    for token in tokens:

        # Remove generic/legal words
        if token in COMMON_NAME_TOKENS:
            continue

        # Ignore very short name tokens
        if len(token) < 4:
            continue

        useful_tokens.append(token)

    return useful_tokens


# --------------------------------------------------
# Address tokens
# --------------------------------------------------

def get_address_tokens(address):
    """
    Get useful components from a business address.

    Keeps:
    - meaningful words
    - numeric components, including 1-digit
      building numbers

    Removes:
    - generic address words
    """

    normalized = normalize_address(address)

    tokens = normalized.split()

    useful_tokens = []

    for token in tokens:

        # Remove generic address words
        if token in COMMON_ADDRESS_TOKENS:
            continue

        # Keep numeric tokens, even 1-digit numbers
        if token.isdigit():
            useful_tokens.append(token)
            continue

        # Keep meaningful words
        if len(token) >= 4:
            useful_tokens.append(token)

    return useful_tokens


# --------------------------------------------------
# Create blocking indexes
# --------------------------------------------------

def create_name_block_index(source_df):
    """
    Create three blocking indexes:

    1. Exact normalized name
    2. Informative name token
    3. Informative address token
    """

    exact_index = {}

    name_token_index = {}

    address_token_index = {}

    for _, row in source_df.iterrows():

        entity_id = row["entity_id"]

        country = str(
            row["country"]
        ).strip().lower()

        # ------------------------------------------
        # Block 1:
        # Country + exact normalized name
        # ------------------------------------------

        normalized_name = normalize_name(
            row["business_name"]
        )

        if normalized_name:

            key = (
                country,
                normalized_name
            )

            if key not in exact_index:

                exact_index[key] = []

            exact_index[key].append(
                entity_id
            )

        # ------------------------------------------
        # Block 2:
        # Country + informative name token
        # ------------------------------------------

        name_tokens = get_name_tokens(
            row["business_name"]
        )

        for token in name_tokens:

            key = (
                country,
                token
            )

            if key not in name_token_index:

                name_token_index[key] = []

            name_token_index[key].append(
                entity_id
            )

        # ------------------------------------------
        # Block 3:
        # Country + informative address token
        # ------------------------------------------

        address_tokens = get_address_tokens(
            row["business_address"]
        )

        for token in address_tokens:

            key = (
                country,
                token
            )

            if key not in address_token_index:

                address_token_index[key] = []

            address_token_index[key].append(
                entity_id
            )

    return (
        exact_index,
        name_token_index,
        address_token_index
    )


# --------------------------------------------------
# Generate candidates
# --------------------------------------------------

def get_candidates(
    s1_row,
    source2_indexes,
    source3_indexes
):
    """
    Generate candidates using three blocking strategies:

    Block 1:
        Exact normalized name

    Block 2:
        Informative name token

    Block 3:
        At least TWO shared informative
        address tokens
    """

    country = str(
        s1_row["country"]
    ).strip().lower()

    normalized_name = normalize_name(
        s1_row["business_name"]
    )

    candidates = set()

    # Unpack S2 indexes
    (
        s2_exact,
        s2_name_tokens,
        s2_address_tokens
    ) = source2_indexes

    # Unpack S3 indexes
    (
        s3_exact,
        s3_name_tokens,
        s3_address_tokens
    ) = source3_indexes

    # ------------------------------------------
    # Block 1: Exact name
    # ------------------------------------------

    if normalized_name:

        key = (
            country,
            normalized_name
        )

        candidates.update(
            s2_exact.get(key, [])
        )

        candidates.update(
            s3_exact.get(key, [])
        )

    # ------------------------------------------
    # Block 2: Name tokens
    # ------------------------------------------

    name_tokens = get_name_tokens(
        s1_row["business_name"]
    )

    for token in name_tokens:

        key = (
            country,
            token
        )

        candidates.update(
            s2_name_tokens.get(key, [])
        )

        candidates.update(
            s3_name_tokens.get(key, [])
        )

    # ------------------------------------------
    # Block 3: Address tokens
    #
    # Require at least TWO shared
    # informative address tokens.
    #
    # A single shared number is NOT enough.
    # ------------------------------------------

    s1_address_tokens = set(
        get_address_tokens(
            s1_row["business_address"]
        )
    )

    if len(s1_address_tokens) >= 2:

        address_candidate_counts = {}

        # --------------------------------------
        # Check S2
        # --------------------------------------

        for token in s1_address_tokens:

            key = (
                country,
                token
            )

            for entity_id in s2_address_tokens.get(
                key,
                []
            ):

                address_candidate_counts[
                    entity_id
                ] = address_candidate_counts.get(
                    entity_id,
                    0
                ) + 1

        # --------------------------------------
        # Check S3
        # --------------------------------------

        for token in s1_address_tokens:

            key = (
                country,
                token
            )

            for entity_id in s3_address_tokens.get(
                key,
                []
            ):

                address_candidate_counts[
                    entity_id
                ] = address_candidate_counts.get(
                    entity_id,
                    0
                ) + 1

        # --------------------------------------
        # Keep candidates sharing 2+ tokens
        # --------------------------------------

        for entity_id, count in address_candidate_counts.items():

            if count >= 2:

                candidates.add(
                    entity_id
                )

    return list(candidates)