import os
import re
from glob import glob
from collections import defaultdict

# Match parameters in the filename
PARAM_KEYS = ["beta", "m", "a", "b", "alpha", "mu", "nu", "gamma"]
CHI_PATTERN = re.compile(r"chi(\d+)")
PARAM_PATTERN = re.compile(r"(" + "|".join(PARAM_KEYS) + r")-?([\d\.]+)")

# All PNG images
images = sorted(glob("*.png"))

# Dictionary: param_name → list of (filename, chi)
grouped = defaultdict(list)

for img in images:
    # Find chi
    chi_match = CHI_PATTERN.search(img)
    chi = chi_match.group(1) if chi_match else "?"

    # Find which parameter this image corresponds to
    for param, val in PARAM_PATTERN.findall(img):
        grouped[param].append((img, chi))
        break  # Only group by the first found param

# HTML start
html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Simulation Results</title>
  <style>
    body { font-family: sans-serif; padding: 20px; background: #f5f5f5; }
    h1 { margin-bottom: 40px; }
    .group { margin-bottom: 50px; }
    .param-title { font-size: 1.5em; margin-bottom: 10px; }
    .chi-group { margin: 10px 0 20px; }
    .chi-title { font-weight: bold; margin-bottom: 5px; }
    .image-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .image-grid img {
      max-width: 300px;
      height: auto;
      border-radius: 6px;
      border: 1px solid #ccc;
    }
  </style>
</head>
<body>

<h1>Simulation Results Grouped by Parameter</h1>
"""

# Build HTML sections
for param in sorted(grouped.keys()):
    html += f"<div class='group'><div class='param-title'>Parameter: <code>{param}</code></div>"

    # Group images by chi
    chi_dict = defaultdict(list)
    for img, chi in grouped[param]:
        chi_dict[chi].append(img)

    for chi in sorted(chi_dict.keys(), key=lambda x: int(x)):
        html += f"<div class='chi-group'><div class='chi-title'>chi = {chi}</div><div class='image-grid'>\n"
        for img in sorted(chi_dict[chi]):
            html += f"<img src='{img}' alt='{img}'>\n"
        html += "</div></div>\n"

    html += "</div>\n"

# Finish HTML
html += "</body></html>"

# Write to index.html
with open("index.html", "w") as f:
    f.write(html)

print("Generated index.html with images grouped by parameter.")
