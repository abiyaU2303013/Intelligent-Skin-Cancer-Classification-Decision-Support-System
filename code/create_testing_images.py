import os
import shutil
import pandas as pd


# =========================================================
# 1. PROJECT PATH
# =========================================================

code_path = os.path.dirname(
    os.path.abspath(__file__)
)

project_path = os.path.dirname(code_path)


# =========================================================
# 2. PATHS
# =========================================================

test_csv = os.path.join(
    project_path,
    "results",
    "test.csv"
)

testing_folder = os.path.join(
    project_path,
    "testing_images"
)

os.makedirs(
    testing_folder,
    exist_ok=True
)


# =========================================================
# 3. LOAD TEST CSV
# =========================================================

test_df = pd.read_csv(test_csv)

print("=" * 60)
print("CREATING TESTING IMAGES FOLDER")
print("=" * 60)

print("\nNumber of test images:", len(test_df))


# =========================================================
# 4. COPY TEST IMAGES
# =========================================================

copied = 0
missing = 0

for _, row in test_df.iterrows():

    image_id = row["image_id"]

    image_name = image_id + ".jpg"

    # Check Part 1
    part1_path = os.path.join(
        project_path,
        "HAM10000_images_part_1",
        image_name
    )

    # Check Part 2
    part2_path = os.path.join(
        project_path,
        "HAM10000_images_part_2",
        image_name
    )

    # Find image
    if os.path.exists(part1_path):

        source_path = part1_path

    elif os.path.exists(part2_path):

        source_path = part2_path

    else:

        print(
            "Missing:",
            image_name
        )

        missing += 1

        continue

    # Destination
    destination_path = os.path.join(
        testing_folder,
        image_name
    )

    shutil.copy2(
        source_path,
        destination_path
    )

    copied += 1


# =========================================================
# 5. FINAL RESULT
# =========================================================

print("\n" + "=" * 60)

print("TESTING IMAGES CREATED")
print("=" * 60)

print("\nImages copied :", copied)

print("Images missing:", missing)

print("\nTesting folder:")

print(testing_folder)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)