#!/usr/bin/env python3
"""
Realistic benchmark data generator for Amazon ML Challenge 2026.
Generates train and test datasets matching the exact format, noise profiles,
and country distributions described in the problem statement.
"""

import os
import random
import csv

random.seed(42)

BUSINESS_BASES_US = [
    ("Acme Robotics", "500 Market St, Suite 400, San Jose, CA 95113"),
    ("Delta Foods", "8 Oak Avenue, Austin, TX 78701"),
    ("Bright Cafe", "22 Pine Street, Reno, NV 89501"),
    ("Zen Traders", "4 Hill Road, Boise, ID 83702"),
    ("Nexus Logistics", "1200 Industrial Parkway, Chicago, IL 60607"),
    ("Apex Healthcare", "742 Evergreen Terrace, Springfield, OR 97477"),
    ("Summit Financial Partners", "100 Wall Street, 15th Floor, New York, NY 10005"),
    ("Horizon Renewable Tech", "350 Boulder Blvd, Boulder, CO 80301"),
    ("Vanguard Auto Parts", "88 Motor Mile Drive, Detroit, MI 48202"),
    ("Blue Ocean Marine", "14 Harbor Way, Seattle, WA 98101"),
    ("Precision Engineering Labs", "55 Technology Square, Cambridge, MA 02139"),
    ("Crestview Real Estate", "670 Sunset Plaza, Los Angeles, CA 90028"),
    ("Ironclad Security Systems", "910 Liberty Avenue, Pittsburgh, PA 15222"),
    ("Silverline Media Productions", "450 Broadway, Floor 8, New York, NY 10013"),
    ("Titan Heavy Industries", "210 Foundry Lane, Cleveland, OH 44114"),
]

BUSINESS_BASES_IN = [
    ("Tata Consultancy Services", "BBD Marg, Fort, Mumbai, Maharashtra 400001"),
    ("Infosys Technologies", "Electronics City, Hosur Road, Bengaluru, Karnataka 560100"),
    ("Reliance Retail Ventures", "Maker Chambers IV, Nariman Point, Mumbai, Maharashtra 400021"),
    ("Wipro Enterprises", "Doddakannelli, Sarjapur Road, Bengaluru, Karnataka 560035"),
    ("Apollo Hospitals Enterprise", "Greams Road, Thousand Lights, Chennai, Tamil Nadu 600006"),
    ("Mahindra and Mahindra", "Gateway Building, Apollo Bunder, Mumbai, Maharashtra 400001"),
    ("Zomato Media", "Ground Floor, Tower C, Pioneer Urban Square, Sector 62, Gurugram, Haryana 122098"),
    ("Swiggy Bundl Technologies", "Maruthi Chambers, Survey 17/9B, Roopena Agrahara, Bengaluru, Karnataka 560068"),
    ("HDFC Bank Financial Services", "HDFC Bank House, Senapati Bapat Marg, Lower Parel, Mumbai, Maharashtra 400013"),
    ("Bharti Airtel Telecom", "Airtel Centre, Plot No 16, Udyog Vihar Phase IV, Gurugram, Haryana 122015"),
    ("Adani Enterprises Limited", "Adani Corporate House, Shantigram, SG Highway, Ahmedabad, Gujarat 382421"),
    ("Larsen and Toubro Construction", "Mount Poonamallee Road, Manapakkam, Chennai, Tamil Nadu 600089"),
    ("Asian Paints Consumer", "6A Shantiniketan, 83 CST Road, Kalina, Santacruz East, Mumbai, Maharashtra 400098"),
    ("Bajaj Finance Solutions", "Mumbai-Pune Road, Akurdi, Pune, Maharashtra 411035"),
    ("Biocon Biologicals", "20th KM, Hosur Road, Electronic City, Bengaluru, Karnataka 560100"),
]

BUSINESS_BASES_FR = [
    ("Schneider Electric Industries", "35 Rue Joseph Monier, 92500 Rueil-Malmaison"),
    ("Air Liquide International", "75 Quai d'Orsay, 75007 Paris"),
    ("Danone Produits Frais", "17 Boulevard Haussmann, 75009 Paris"),
    ("Michelin Pneumatiques", "23 Place des Carmes Dechaux, 63000 Clermont-Ferrand"),
    ("Capgemini Services", "11 Rue de Tilsitt, 75017 Paris"),
    ("L'Oreal Cosmetiques", "41 Rue Martre, 92117 Clichy"),
    ("TotalEnergies Renouvelables", "2 Place Jean Millier, La Defense 6, 92400 Courbevoie"),
    ("Sanofi Pasteur Sante", "54 Rue La Boetie, 75008 Paris"),
]

LEGAL_SUFFIXES_MAP = {
    "Inc.": ["Incorporated", "Inc", "", "Corp."],
    "Corp": ["Corporation", "Corp.", "", "Inc."],
    "Pvt Ltd": ["Private Limited", "Pvt. Ltd.", "Limited", "Ltd."],
    "LLC": ["L.L.C.", "Limited Liability Co", "", "Company"],
    "Limited": ["Ltd", "Ltd.", "Enterprises"],
}

ADDRESS_ABBREV_MAP = {
    "Street": "St.",
    "Road": "Rd.",
    "Avenue": "Ave.",
    "Boulevard": "Blvd.",
    "Drive": "Dr.",
    "Lane": "Ln.",
    "Floor": "Fl.",
    "Suite": "Ste.",
}

LANDMARKS_IN = [
    "Near SBI ATM", "Opp Metro Station", "Behind Bus Depot", "Beside City Center Mall",
    "Near Petrol Pump", "2nd Cross", "Opposite Post Office"
]

LANDMARKS_US = [
    "Nr City Hall", "Behind Central Library", "Near Grand Plaza", "Adjacent to Station 4",
    "Suite 200", "Corner of 5th Ave"
]

def corrupt_name(name, country):
    words = name.split()
    if random.random() < 0.4 and "&" in name:
        name = name.replace("&", "and")
    elif random.random() < 0.4 and "and" in name:
        name = name.replace("and", "&")

    # Legal suffix variation
    for k, replacements in LEGAL_SUFFIXES_MAP.items():
        if k in name:
            rep = random.choice(replacements)
            name = name.replace(k, rep).strip()
            break
    else:
        if random.random() < 0.5:
            if country == "India":
                name = name + " " + random.choice(["Pvt Ltd", "Private Limited", "Enterprises", "India"])
            elif country == "US":
                name = name + " " + random.choice(["Inc", "LLC", "Corp", "USA"])
            elif country == "France":
                name = name + " " + random.choice(["SARL", "SAS", "France", "SA"])

    # Occasional small typo (1 char swap or drop)
    if random.random() < 0.25 and len(name) > 6:
        idx = random.randint(2, len(name) - 3)
        name = name[:idx] + name[idx+1:]

    return " ".join(name.split())

def corrupt_address(addr, country):
    # Abbreviation expansion/contraction
    for full, abbrev in ADDRESS_ABBREV_MAP.items():
        if full in addr and random.random() < 0.6:
            addr = addr.replace(full, abbrev)
        elif abbrev in addr and random.random() < 0.6:
            addr = addr.replace(abbrev, full)

    # Landmark insertion
    if random.random() < 0.35:
        if country == "India":
            landmark = random.choice(LANDMARKS_IN)
            addr = f"{landmark}, {addr}"
        elif country == "US":
            landmark = random.choice(LANDMARKS_US)
            addr = f"{addr}, {landmark}"

    # Missing components (e.g. drop postal code or state)
    parts = [p.strip() for p in addr.split(",")]
    if len(parts) > 2 and random.random() < 0.3:
        parts.pop()
        addr = ", ".join(parts)

    return addr

def generate_dataset(num_entities, countries, start_id, is_test=False):
    s1_records = []
    s2_records = []
    s3_records = []
    ground_truth = [] # (s1_id, [matched_ids])

    s2_id_counter = 1
    s3_id_counter = 1

    for i in range(num_entities):
        s1_id = f"S1-{start_id + i:05d}"
        country = random.choice(countries)
        if country == "US":
            base_name, base_addr = random.choice(BUSINESS_BASES_US)
        elif country == "India":
            base_name, base_addr = random.choice(BUSINESS_BASES_IN)
        else: # France
            base_name, base_addr = random.choice(BUSINESS_BASES_FR)

        # Distinguish slightly for unique reference
        s1_name = f"{base_name} {start_id + i}" if i > 15 else base_name
        s1_addr = f"{random.randint(10, 999)} {base_addr}" if i > 15 else base_addr
        s1_records.append((s1_id, s1_name, s1_addr, country))

        matched_ids = []

        # 20% singletons (no match in S2 or S3)
        is_singleton = random.random() < 0.20
        if not is_singleton:
            # 70% chance of S2 match
            if random.random() < 0.70:
                s2_id = f"S2-{s2_id_counter:05d}"
                s2_id_counter += 1
                s2_name = corrupt_name(s1_name, country)
                s2_addr = corrupt_address(s1_addr, country)
                s2_records.append((s2_id, s2_name, s2_addr, country))
                matched_ids.append(s2_id)

            # 60% chance of S3 match
            if random.random() < 0.60:
                s3_id = f"S3-{s3_id_counter:05d}"
                s3_id_counter += 1
                s3_name = corrupt_name(s1_name, country)
                s3_addr = corrupt_address(s1_addr, country)
                s3_records.append((s3_id, s3_name, s3_addr, country))
                matched_ids.append(s3_id)

            # Possible 2nd match in S2 (e.g., duplicate branch/registration)
            if random.random() < 0.15:
                s2_id_extra = f"S2-{s2_id_counter:05d}"
                s2_id_counter += 1
                s2_name = corrupt_name(s1_name, country)
                s2_addr = corrupt_address(s1_addr, country)
                s2_records.append((s2_id_extra, s2_name, s2_addr, country))
                matched_ids.append(s2_id_extra)

        ground_truth.append((s1_id, ",".join(matched_ids)))

    # Add distractors / unrelated records to S2 and S3 (noise records with NO S1 match)
    for _ in range(num_entities // 4):
        c = random.choice(countries)
        s2_id = f"S2-{s2_id_counter:05d}"
        s2_id_counter += 1
        s2_records.append((s2_id, f"Unrelated Business {s2_id}", f"100 Unknown Street, Metropolis", c))

    for _ in range(num_entities // 4):
        c = random.choice(countries)
        s3_id = f"S3-{s3_id_counter:05d}"
        s3_id_counter += 1
        s3_records.append((s3_id, f"Random Commercial Org {s3_id}", f"500 Bypass Highway, Gotham", c))

    # Shuffle S2 and S3 to simulate real unlinked lists
    random.shuffle(s2_records)
    random.shuffle(s3_records)

    return s1_records, s2_records, s3_records, ground_truth

def write_tsv(filepath, header, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f"Created {filepath} ({len(rows)} records)")

def main():
    print("Generating training dataset (US and India)...")
    s1_train, s2_train, s3_train, gt_train = generate_dataset(
        num_entities=150,
        countries=["US", "India"],
        start_id=1,
        is_test=False
    )
    write_tsv("dataset/train/train_source1.tsv", ["entity_id", "business_name", "business_address", "country"], s1_train)
    write_tsv("dataset/train/train_source2.tsv", ["entity_id", "business_name", "business_address", "country"], s2_train)
    write_tsv("dataset/train/train_source3.tsv", ["entity_id", "business_name", "business_address", "country"], s3_train)
    write_tsv("dataset/train/train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gt_train)

    print("\nGenerating test dataset (US, India, and France)...")
    s1_test, s2_test, s3_test, _ = generate_dataset(
        num_entities=100,
        countries=["US", "India", "France"],
        start_id=10001,
        is_test=True
    )
    write_tsv("dataset/test/test_source1.tsv", ["entity_id", "business_name", "business_address", "country"], s1_test)
    write_tsv("dataset/test/test_source2.tsv", ["entity_id", "business_name", "business_address", "country"], s2_test)
    write_tsv("dataset/test/test_source3.tsv", ["entity_id", "business_name", "business_address", "country"], s3_test)
    print("\nMock benchmark datasets generated successfully!")

if __name__ == "__main__":
    main()
